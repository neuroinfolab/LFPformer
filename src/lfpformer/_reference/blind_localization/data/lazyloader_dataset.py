import os
import sys
import pickle
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

# Shim: pickles built with newer numpy (numpy._core) fail in containers with
# older numpy (numpy.core) and vice versa.  Map the missing module so
# pickle.load can find it regardless of which numpy is installed.
if not hasattr(np, "_core"):
    import numpy.core as _nc
    sys.modules["numpy._core"] = _nc
    sys.modules["numpy._core.multiarray"] = getattr(_nc, "multiarray", _nc)
elif not hasattr(np, "core"):
    import numpy._core as _nc
    sys.modules["numpy.core"] = _nc
    sys.modules["numpy.core.multiarray"] = getattr(_nc, "multiarray", _nc)
import torch
from torch.utils.data import Dataset

# scipy.signal imported lazily below to avoid crash on import in Singularity


class CompactDataset(Dataset):
    """
    Dataset for reading pre-extracted compact balanced data.

    The compact format stores samples in small pickle files (~200 samples each):
    - compact_index.pkl: metadata with file_list and cumulative_counts
    - samples_XXXX.pkl: list of (signal, meta_str, label_id[, coords]) tuples

    This format is ~7000x smaller than the original session pickles and can
    be staged to local SSD ($SLURM_TMPDIR) for fast I/O.
    """

    def __init__(
        self,
        compact_dir: str,
        include_labels: bool = False,
        target_sampling_rate: int = 16000,
        super_regions: bool = False,
        dataset: str = "Allen",
        already_resampled: bool = False,
        gpu_resample: bool = False,
        input_sr: Optional[int] = None,
        model_sr: Optional[int] = None,
        verbose: bool = True,
        return_region_id: bool = False,
        tile_to_length: Optional[int] = None,
        atlas_depth: Optional[int] = None,
    ):
        super().__init__()

        self.compact_dir = Path(compact_dir)
        self.include_labels = bool(include_labels)
        self.super_regions = bool(super_regions)
        self.dataset = str(dataset)
        self.verbose = bool(verbose)
        self.return_region_id = bool(return_region_id)
        self.tile_to_length = tile_to_length
        self.atlas_depth = atlas_depth

        # Sampling rate handling
        if model_sr is not None:
            target_sampling_rate = int(model_sr)
        self.target_sampling_rate = int(target_sampling_rate)

        default_input_sr = 2500 if dataset == "Monkey" else 1250
        if input_sr is not None:
            default_input_sr = int(input_sr)

        self.already_resampled = bool(already_resampled)
        self.gpu_resample = bool(gpu_resample)

        if self.already_resampled:
            self.signal_sampling_rate = self.target_sampling_rate
        else:
            self.signal_sampling_rate = int(default_input_sr)

        # Atlas metadata (label_maps by depth) — loaded from parent dir's metadata.pkl
        self._atlas_label_maps: Optional[Dict] = None
        self._atlas_metadata: Optional[Dict] = None
        self._global_effective_maps: Optional[Dict[int, Dict[str, int]]] = None
        self._global_label_map_source: Optional[str] = None
        self._cumulative_label_map: Optional[Dict[str, int]] = None
        self._effective_label_map: Optional[Dict[str, int]] = None
        self._effective_id_to_name: Optional[Dict[int, str]] = None
        if self.atlas_depth is not None:
            self._load_atlas_metadata()
            self._cumulative_label_map = self._build_cumulative_label_map(self.atlas_depth)

        # Label mappings (same as SessionDataset) — used when atlas_depth is None
        if dataset != "Monkey":
            self._coarse_label_map = {
                'VIS': 0, 'HPF': 1, 'OLF': 2, 'AMY': 3, 'CTX-STR': 4, 'TH-GEN': 5, 'VIS-TH': 6,
                'AUD-TH': 7, 'SM-TH': 8, 'RT-ZI': 9, 'MB-Tect': 10, 'MB-Teg': 11, 'Misc': 12
            }
        else:
            self._coarse_label_map = {'imec0': 0, 'imec1': 1, 'imec2': 2}

        self._id_to_coarse = {v: k for k, v in self._coarse_label_map.items()}

        # GCP full-recording memmap layout (see gcp_neuroframe/GCP_PRETRAINING.md §2f):
        # O(1) random access, no per-worker shard cache, no full scan at init.
        self._memmap = (self.compact_dir / "memmap_index.json").exists()
        if self._memmap:
            self._init_memmap()
            self._build_region_dict()
            return

        # Load compact index
        index_path = self.compact_dir / "compact_index.pkl"
        if not index_path.exists():
            raise FileNotFoundError(f"Compact index not found: {index_path}")

        with open(index_path, "rb") as f:
            self.index = pickle.load(f)

        self.n_samples = self.index["n_samples"]
        self.file_list = self.index["file_list"]
        self.cumulative_counts = self.index["cumulative_counts"]

        # Cache for loaded sample files (they're small, cache all)
        self._file_cache: Dict[int, List[Tuple]] = {}

        # Build region dict from first file
        self._build_region_dict()

        if self.verbose:
            print(f"[CompactDataset] Loaded {self.n_samples} samples from {len(self.file_list)} files")
            print(f"[CompactDataset] Directory: {self.compact_dir}")
            if self.atlas_depth is not None and self._cumulative_label_map is not None:
                n_cum = len(self._cumulative_label_map)
                n_eff = len(self._effective_label_map) if self._effective_label_map else n_cum
                print(f"[CompactDataset] Atlas depth={self.atlas_depth} (cumulative 0..{self.atlas_depth}): "
                      f"{n_cum} cumulative classes, {n_eff} effective (with samples)")
                if self._atlas_label_maps:
                    for d in range(self.atlas_depth + 1):
                        d_map = self._atlas_label_maps.get(d, {})
                        print(f"  depth {d}: {len(d_map)} classes")
            elif self.atlas_depth is not None:
                n_cls = len(self._atlas_label_maps.get(self.atlas_depth, {})) if self._atlas_label_maps else 0
                print(f"[CompactDataset] Atlas depth={self.atlas_depth}, {n_cls} classes")

    def _load_atlas_metadata(self):
        """Load atlas metadata.pkl from the dataset's parent directory."""
        # metadata.pkl lives one level above train/val/test
        meta_path = self.compact_dir.parent / "metadata.pkl"
        if not meta_path.exists():
            # Also check in compact_dir itself (flat layout)
            meta_path = self.compact_dir / "metadata.pkl"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Atlas metadata not found at {self.compact_dir.parent / 'metadata.pkl'} "
                f"or {self.compact_dir / 'metadata.pkl'}"
            )
        with open(meta_path, "rb") as f:
            try:
                self._atlas_metadata = pickle.load(f)
            except ModuleNotFoundError:
                # numpy._core vs numpy.core mismatch (login node vs container)
                f.seek(0)
                class _NumpyCompatUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module.startswith("numpy._core"):
                            module = module.replace("numpy._core", "numpy.core", 1)
                        return super().find_class(module, name)
                self._atlas_metadata = _NumpyCompatUnpickler(f).load()
        self._atlas_label_maps = self._atlas_metadata.get("label_maps", {})
        # Global effective label maps (built from all splits at dataset build time)
        self._global_effective_maps = self._atlas_metadata.get("effective_label_maps_by_depth")
        self._global_label_map_source = self._atlas_metadata.get("global_label_map_source")
        if self.verbose:
            depths = sorted(k for k in self._atlas_label_maps if isinstance(k, int))
            has_global = self._global_effective_maps is not None
            print(f"[CompactDataset] Loaded atlas metadata: depths {depths}, "
                  f"global_effective_maps={has_global}, source={self._global_label_map_source}")

    def _build_cumulative_label_map(self, depth: int) -> Dict[str, int]:
        """Build label map as union of all label_maps from depth 0..depth."""
        cumulative: Dict[str, int] = {}
        idx = 0
        for d in range(0, depth + 1):
            for name in sorted(self._atlas_label_maps.get(d, {}).keys()):
                if name not in cumulative:
                    cumulative[name] = idx
                    idx += 1
        return cumulative

    @property
    def cumulative_label_map(self) -> Optional[Dict[str, int]]:
        """Returns the cumulative label map (union of depths 0..atlas_depth)."""
        return self._cumulative_label_map

    @property
    def effective_label_map(self) -> Optional[Dict[str, int]]:
        """Returns the global intersection label map (contiguous 0..N-1).

        Built as the intersection of regions across ALL datasets and ALL splits,
        ensuring consistent class IDs everywhere. Loaded from metadata.pkl
        (key: effective_label_maps_by_depth).
        """
        return self._effective_label_map

    @property
    def effective_id_to_name(self) -> Optional[Dict[int, str]]:
        """Returns mapping from effective label id to region name."""
        return self._effective_id_to_name

    @property
    def global_label_map_source(self) -> Optional[str]:
        """Returns the source file of the global label map (e.g. 'global_effective_label_maps.pkl')."""
        return self._global_label_map_source

    @property
    def num_effective_classes(self) -> int:
        """Number of classes in the global intersection label map."""
        if self._effective_label_map is not None:
            return len(self._effective_label_map)
        return len(self._coarse_label_map)

    def set_atlas_depth(self, depth: int):
        """Switch atlas depth without reloading data. Rebuilds cumulative map and region_dict."""
        if self._atlas_label_maps is None:
            self._load_atlas_metadata()
        self.atlas_depth = depth
        self._cumulative_label_map = self._build_cumulative_label_map(depth)
        self._build_region_dict()

    def restrict_to_labels(self, valid_label_ids: set):
        """Restrict the effective label map to only the given label IDs.

        Builds a new contiguous label map (0..N-1) from the subset of labels
        that actually appear in the data across all splits.  After calling this,
        __getitem__ returns -1 for any sample whose original label is not in
        *valid_label_ids*, so the dataloader / extraction code will skip it.

        Args:
            valid_label_ids: set of original integer label IDs to keep.
        """
        if self._effective_label_map is None:
            return
        # Build old_id -> name for the kept classes
        old_id_to_name = {v: k for k, v in self._effective_label_map.items()}
        kept = sorted(valid_label_ids)
        # New contiguous map: name -> new_id  (0..N-1)
        new_map = {}
        for new_id, old_id in enumerate(kept):
            name = old_id_to_name.get(old_id)
            if name is not None:
                new_map[name] = new_id
        self._effective_label_map = new_map
        self._effective_id_to_name = {v: k for k, v in new_map.items()}
        # Also store the remap table so _get_atlas_label_id works via the new map
        # (it looks up region_name in _effective_label_map, which now has new ids)

    def _get_atlas_label_id(self, hierarchy_dict: Dict, depth: int) -> int:
        """Map atlas hierarchy dict to integer label using the global intersection map.

        Uses the deepest available label at min(depth, max_depth), looked up
        in the global intersection label map (contiguous 0..N-1).
        Falls back to cumulative label map if effective map is not yet built.
        Returns -1 if the region is not in the intersection map.
        """
        max_d = hierarchy_dict.get('max_depth', depth)
        effective_depth = min(depth, max_d)
        region_name = hierarchy_dict.get(effective_depth)
        if region_name is None:
            return -1
        # Prefer effective map (contiguous); fall back to cumulative during build
        label_map = self._effective_label_map or self._cumulative_label_map
        if label_map is None:
            return -1
        return label_map.get(region_name, -1)

    # ---- memmap layout -------------------------------------------------------
    def _init_memmap(self):
        import json
        idx = json.load(open(self.compact_dir / "memmap_index.json"))
        self.index = idx
        self.n_samples = int(idx["n_samples"])
        self._mm_chunk_len = int(idx.get("chunk_len", 3750))
        self._mm_names = [p["name"] for p in idx["parts"]]
        counts = np.array([int(p["n"]) for p in idx["parts"]], dtype=np.int64)
        self._mm_starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        self._mm_counts = counts
        with open(self.compact_dir / "hier_by_fine_id.pkl", "rb") as f:
            self._mm_hier = pickle.load(f)
        # small per-sample arrays are loaded fully (fine_id is needed for region counts)
        self._mm_fine = np.concatenate([
            np.fromfile(self.compact_dir / "parts" / nm / "fine_id.i32", dtype=np.int32)
            for nm in self._mm_names]) if self._mm_names else np.zeros(0, np.int32)
        assert len(self._mm_fine) == self.n_samples, (len(self._mm_fine), self.n_samples)
        self._mm_open: Dict[int, Tuple] = {}  # per-process lazily opened memmaps
        self.file_list = []
        self.cumulative_counts = self._mm_starts
        self._file_cache = {}

    def _mm_part(self, pi: int):
        h = self._mm_open.get(pi)
        if h is None:
            d = self.compact_dir / "parts" / self._mm_names[pi]
            n = int(self._mm_counts[pi])
            sig = np.memmap(d / "signals.f32", dtype=np.float32, mode="r", shape=(n, self._mm_chunk_len))
            crd = np.fromfile(d / "coords.f32", dtype=np.float32).reshape(n, 3)
            h = (sig, crd)
            self._mm_open[pi] = h
        return h

    def _mm_locate(self, idx: int) -> Tuple[int, int]:
        pi = int(np.searchsorted(self._mm_starts, idx, side="right") - 1)
        return pi, int(idx - self._mm_starts[pi])

    def _mm_sample(self, idx: int):
        pi, off = self._mm_locate(idx)
        sig, crd = self._mm_part(pi)
        return np.array(sig[off]), self._mm_hier[int(self._mm_fine[idx])], crd[off]

    def __getstate__(self):
        # don't ship open memmaps to DataLoader workers; they reopen lazily
        st = self.__dict__.copy()
        if st.get("_memmap"):
            st["_mm_open"] = {}
        return st

    def _build_region_dict(self):
        """Build region distribution by scanning all samples (cumulative depth resolution)."""
        region_counts: Dict[str, int] = {}

        if getattr(self, "_memmap", False):
            # Same counting rule as the pickle path, vectorized over fine_id.
            fids, cnts = np.unique(self._mm_fine, return_counts=True)
            for fid, c in zip(fids.tolist(), cnts.tolist()):
                raw_label = self._mm_hier[int(fid)]
                if self.atlas_depth is not None:
                    max_d = raw_label.get('max_depth', self.atlas_depth)
                    effective_depth = min(self.atlas_depth, max_d)
                    region_name = raw_label.get(effective_depth)
                    if region_name is None:
                        continue
                    region_counts[region_name] = region_counts.get(region_name, 0) + int(c)

        for file_idx in range(len(self.file_list)):
            samples = self._load_file(file_idx)
            for sample in samples:
                raw_label = sample[2]
                if isinstance(raw_label, dict) and self.atlas_depth is not None:
                    # Cumulative: use deepest available label ≤ atlas_depth
                    max_d = raw_label.get('max_depth', self.atlas_depth)
                    effective_depth = min(self.atlas_depth, max_d)
                    region_name = raw_label.get(effective_depth)
                    if region_name is None:
                        continue
                    region_counts[region_name] = region_counts.get(region_name, 0) + 1
                elif not isinstance(raw_label, dict):
                    label_id = raw_label
                    label_name = self._id_to_coarse.get(label_id, f"class_{label_id}")
                    region_counts[label_name] = region_counts.get(label_name, 0) + 1
                # else: dict label with no atlas_depth — skip

        # Build effective label remap: contiguous 0..N-1 for regions with samples
        if self.atlas_depth is not None and self._cumulative_label_map is not None:
            if (self._global_effective_maps is not None
                    and self.atlas_depth in self._global_effective_maps):
                self._effective_label_map = dict(self._global_effective_maps[self.atlas_depth])
            else:
                raise RuntimeError(
                    f"Global effective label map not found for atlas_depth={self.atlas_depth}. "
                    f"Run script/build_global_label_map.py to add effective_label_maps_by_depth "
                    f"to metadata.pkl."
                )
            self._effective_id_to_name = {idx: name for name, idx in self._effective_label_map.items()}
        else:
            self._effective_label_map = None
            self._effective_id_to_name = None

        self.region_dict = region_counts
        total = sum(region_counts.values()) if region_counts else 0
        most_common = max(region_counts.values()) if region_counts else 0
        self.chance_accuracy = (most_common / total) if total > 0 else 0.0

    def _load_file(self, file_idx: int) -> List[Tuple]:
        """Load a sample file, with caching."""
        if file_idx in self._file_cache:
            return self._file_cache[file_idx]

        # Extract filename from stored path and reconstruct with compact_dir
        # This allows the data to work when copied to a different location (e.g., local SSD)
        stored_path = self.file_list[file_idx]
        filename = Path(stored_path).name  # Just the filename, e.g., "samples_0000.pkl"
        filepath = self.compact_dir / filename

        with open(filepath, "rb") as f:
            samples = pickle.load(f)

        self._file_cache[file_idx] = samples
        return samples

    def _find_file_and_offset(self, idx: int) -> Tuple[int, int]:
        """Find which file contains the given index and the offset within that file."""
        # Binary search in cumulative_counts
        file_idx = np.searchsorted(self.cumulative_counts, idx + 1) - 1
        offset = idx - self.cumulative_counts[file_idx]
        return int(file_idx), int(offset)

    def __len__(self) -> int:
        return self.n_samples

    def get_hierarchy(self, idx: int):
        """Return the raw atlas hierarchy dict for sample idx."""
        if getattr(self, "_memmap", False):
            return self._mm_hier[int(self._mm_fine[idx])]
        file_idx, offset = self._find_file_and_offset(idx)
        samples = self._load_file(file_idx)
        sample = samples[offset]
        raw_label = sample[2]
        return raw_label if isinstance(raw_label, dict) else None

    def __getitem__(self, idx: int):
        if getattr(self, "_memmap", False):
            signal, raw_label, coords = self._mm_sample(int(idx))
        else:
            file_idx, offset = self._find_file_and_offset(idx)
            samples = self._load_file(file_idx)
            sample = samples[offset]
            signal, meta_str, raw_label = sample[0], sample[1], sample[2]
            coords = sample[3] if len(sample) > 3 else (0.0, 0.0, 0.0)

        # Resolve label_id: atlas hierarchy dict → int, or legacy int passthrough
        if isinstance(raw_label, dict) and self.atlas_depth is not None:
            label_id = self._get_atlas_label_id(raw_label, self.atlas_depth)
        elif isinstance(raw_label, dict):
            label_id = -1  # atlas data but no depth specified
        else:
            label_id = raw_label

        # Convert to float32
        x = np.asarray(signal, dtype=np.float32)

        # Optional CPU resample
        if (not self.already_resampled) and (not self.gpu_resample) and (self.signal_sampling_rate != self.target_sampling_rate):
            from scipy.signal import resample
            num_target = int(round(len(x) * self.target_sampling_rate / self.signal_sampling_rate))
            x = resample(x, num_target).astype(np.float32, copy=False)

        # Tile short chunks to target length (e.g. 3s → 30s by repeating 10x)
        if self.tile_to_length is not None and len(x) < self.tile_to_length:
            reps = int(np.ceil(self.tile_to_length / len(x)))
            x = np.tile(x, reps)[:self.tile_to_length]

        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)

        x_t = torch.from_numpy(x)
        coords_t = torch.tensor(coords, dtype=torch.float32)  # [3]

        if not self.include_labels:
            if self.return_region_id:
                return x_t, torch.tensor(label_id, dtype=torch.long), coords_t
            return x_t, torch.tensor(0, dtype=torch.long), coords_t

        return x_t, torch.tensor(label_id, dtype=torch.long), coords_t

    def get_class_distribution(self) -> Dict[str, int]:
        """Return the distribution of classes in this dataset."""
        return dict(self.region_dict) if self.region_dict else {}


class SessionDataset(Dataset):
    """
    SessionDataset with:
      - Index caching (disk + optional mmap)
      - LRU pickle caching
      - Deterministic subsetting
      - Optional CPU resampling (SciPy) OR skip (already_resampled / gpu_resample)
      - Optional pre-built balanced index for HP tuning
    """

    def __init__(
        self,
        session_paths: Optional[List[str]] = None,
        include_labels: bool = False,
        target_sampling_rate: int = 16000,
        cache_size: int = 0,
        data_subset_percentage: float = 1.0,
        super_regions: bool = False,
        dataset: str = "Allen",
        seed: int = 42,
        index_cache_dir: Optional[str] = None,
        mmap_index: bool = True,
        verbose: bool = True,

        # ---- NEW: compatibility with your training script ----
        already_resampled: bool = False,
        gpu_resample: bool = False,
        input_sr: Optional[int] = None,
        model_sr: Optional[int] = None,

        # ---- NEW: pre-built balanced index for HP tuning ----
        balanced_index_path: Optional[str] = None,
    ):
        super().__init__()

        self.balanced_index_path = balanced_index_path
        self.session_paths = list(session_paths) if session_paths else []
        self.include_labels = bool(include_labels)

        # Interpretation:
        # - model_sr: sampling rate the model expects
        # - target_sampling_rate: keep as the dataset's "target" rate (same as model_sr)
        if model_sr is not None:
            target_sampling_rate = int(model_sr)

        self.target_sampling_rate = int(target_sampling_rate)

        # Original sampling rate (Allen typically 1250; Monkey 2500)
        default_input_sr = 2500 if dataset == "Monkey" else 1250
        if input_sr is not None:
            default_input_sr = int(input_sr)

        # If already_resampled=True, the on-disk signal is already at target SR.
        # If gpu_resample=True, we return raw signal at input_sr and resample later on GPU.
        self.already_resampled = bool(already_resampled)
        self.gpu_resample = bool(gpu_resample)

        if self.already_resampled:
            self.signal_sampling_rate = self.target_sampling_rate
        else:
            self.signal_sampling_rate = int(default_input_sr)

        self.data_subset_percentage = float(data_subset_percentage)
        self.super_regions = bool(super_regions)
        self.dataset = str(dataset)
        self.seed = int(seed)
        self.verbose = bool(verbose)

        # LRU cache for loaded session pickles
        self.cache_size = int(cache_size)
        self.cache: "OrderedDict[str, list]" = OrderedDict()

        # -------------------------------
        # Label mappings (unchanged)
        # -------------------------------
        if dataset != "Monkey":
            self._fine_label_map = {
                'APN': 0, 'BMAa': 1, 'CA1': 2, 'CA2': 3, 'CA3': 4, 'COAa': 5, 'COApm': 6, 'CP': 7,
                'DG': 8, 'Eth': 9, 'HPF': 10, 'IGL': 11, 'IntG': 12, 'LD': 13, 'LGd': 14, 'LGv': 15,
                'LP': 16, 'LT': 17, 'MB': 18, 'MGd': 19, 'MGm': 20, 'MGv': 21, 'MRN': 22, 'NOT': 23,
                'OLF': 24, 'OP': 25, 'PF': 26, 'PIL': 27, 'PO': 28, 'POL': 29, 'POST': 30, 'PP': 31,
                'PPT': 32, 'PRE': 33, 'PoT': 34, 'ProS': 35, 'RPF': 36, 'RT': 37, 'SCig': 38,
                'SCiw': 39, 'SCop': 40, 'SCsg': 41, 'SCzo': 42, 'SGN': 43, 'SUB': 44, 'TH': 45,
                'VIS': 46, 'VISal': 47, 'VISam': 48, 'VISl': 49, 'VISli': 50, 'VISmma': 51,
                'VISmmp': 52, 'VISp': 53, 'VISpm': 54, 'VISrl': 55, 'VL': 56, 'VPL': 57, 'VPM': 58,
                'ZI': 59, 'grey': 60, 'na': 61, 'nan': 62
            }
            self._coarse_label_map = {
                'VIS': 0, 'HPF': 1, 'OLF': 2, 'AMY': 3, 'CTX-STR': 4, 'TH-GEN': 5, 'VIS-TH': 6,
                'AUD-TH': 7, 'SM-TH': 8, 'RT-ZI': 9, 'MB-Tect': 10, 'MB-Teg': 11, 'Misc': 12
            }
            self._fine_to_coarse_label_map = {
                'VIS': 'VIS', 'VISal': 'VIS', 'VISam': 'VIS', 'VISl': 'VIS', 'VISli': 'VIS',
                'VISp': 'VIS', 'VISpm': 'VIS', 'VISrl': 'VIS', 'VISmma': 'VIS', 'VISmmp': 'VIS',
                'CA1': 'HPF', 'CA2': 'HPF', 'CA3': 'HPF', 'DG': 'HPF', 'HPF': 'HPF',
                'ProS': 'HPF', 'SUB': 'HPF',
                'OLF': 'OLF', 'Eth': 'OLF',
                'BMAa': 'AMY', 'COAa': 'AMY', 'COApm': 'AMY',
                'CP': 'CTX-STR', 'POST': 'CTX-STR', 'PRE': 'CTX-STR', 'PP': 'CTX-STR',
                'RPF': 'CTX-STR', 'OP': 'CTX-STR', 'POL': 'CTX-STR',
                'TH': 'TH-GEN', 'PF': 'TH-GEN', 'PIL': 'TH-GEN', 'PoT': 'TH-GEN',
                'IGL': 'VIS-TH', 'LD': 'VIS-TH', 'LGd': 'VIS-TH', 'LGv': 'VIS-TH', 'LP': 'VIS-TH',
                'MGd': 'AUD-TH', 'MGm': 'AUD-TH', 'MGv': 'AUD-TH',
                'VPL': 'SM-TH', 'VPM': 'SM-TH', 'PO': 'SM-TH', 'VL': 'SM-TH',
                'RT': 'RT-ZI', 'ZI': 'RT-ZI',
                'APN': 'MB-Tect', 'SCig': 'MB-Tect', 'SCiw': 'MB-Tect', 'SCop': 'MB-Tect',
                'SCsg': 'MB-Tect', 'SCzo': 'MB-Tect', 'NOT': 'MB-Tect',
                'MB': 'MB-Teg', 'MRN': 'MB-Teg', 'PPT': 'MB-Teg', 'SGN': 'MB-Teg', 'LT': 'MB-Teg',
                'IntG': 'Misc'
            }
        else:
            self._fine_label_map = {'imec0': 0, 'imec1': 1, 'imec2': 2}
            self._coarse_label_map = {'imec0': 0, 'imec1': 1, 'imec2': 2}
            self._fine_to_coarse_label_map = {'imec0': 'imec0', 'imec1': 'imec1', 'imec2': 'imec2'}

        # -------------------------------
        # Index cache configuration (unchanged)
        # -------------------------------
        if index_cache_dir is None:
            index_cache_dir = os.environ.get("SESSIONDATASET_INDEX_CACHE_DIR", None)
        self.index_cache_dir = Path(index_cache_dir) if index_cache_dir else None
        self.mmap_index = bool(mmap_index)

        self.session_path_list: List[str] = []
        self.file_ids: np.ndarray
        self.rec_ids: np.ndarray
        self.region_dict: Dict[str, int] = {}
        self.trials: List[str] = []
        self.chance_accuracy: float = 0.0

        self._load_or_build_index()

    # -------------------------------
    # Load pre-built balanced index
    # -------------------------------
    def _load_balanced_index(self) -> bool:
        """
        Load pre-built balanced index if path is provided.
        Returns True if loaded successfully, False otherwise.
        """
        if self.balanced_index_path is None:
            return False

        path = Path(self.balanced_index_path)
        if not path.exists():
            if self.verbose:
                print(f"[SessionDataset] Balanced index not found: {path}")
            return False

        if self.verbose:
            print(f"[SessionDataset] Loading balanced index: {path}")

        with open(path, "rb") as f:
            index_data = pickle.load(f)

        self.session_path_list = index_data["session_path_list"]
        self.file_ids = np.asarray(index_data["file_ids"], dtype=np.int32)
        self.rec_ids = np.asarray(index_data["rec_ids"], dtype=np.int32)
        self.label_ids = np.asarray(index_data["label_ids"], dtype=np.int32)

        # Build region_dict from label_ids
        if self.super_regions:
            coarse_names = {v: k for k, v in self._coarse_label_map.items()}
            unique, counts = np.unique(self.label_ids, return_counts=True)
            self.region_dict = {coarse_names.get(int(l), f"class_{l}"): int(c) for l, c in zip(unique, counts)}
        else:
            fine_names = {v: k for k, v in self._fine_label_map.items()}
            unique, counts = np.unique(self.label_ids, return_counts=True)
            self.region_dict = {fine_names.get(int(l), f"class_{l}"): int(c) for l, c in zip(unique, counts)}

        total = sum(self.region_dict.values()) if self.region_dict else 0
        most_common = max(self.region_dict.values()) if self.region_dict else 0
        self.chance_accuracy = (most_common / total) if total > 0 else 0.0

        # trials not available in balanced index
        self.trials = []

        if self.verbose:
            print(f"[SessionDataset] Loaded {len(self.file_ids)} samples from balanced index")
            print(f"[SessionDataset] Class distribution: {self.region_dict}")

        return True

    # -------------------------------
    # Deterministic subsetting (unchanged)
    # -------------------------------
    def _keep_recording(self, session_path: str, recording_idx: int) -> bool:
        if self.data_subset_percentage >= 1.0:
            return True
        if self.data_subset_percentage <= 0.0:
            return False
        h = hashlib.sha1(f"{self.seed}|{session_path}|{recording_idx}".encode("utf-8")).digest()
        u = int.from_bytes(h[:8], byteorder="big", signed=False)
        x = (u % 10_000_000) / 10_000_000.0
        return x <= self.data_subset_percentage

    def _index_cache_key(self) -> str:
        joined = "\n".join(self.session_paths)
        payload = f"""
paths:
{joined}
include_labels={self.include_labels}
target_sr={self.target_sampling_rate}
signal_sr={self.signal_sampling_rate}
subset={self.data_subset_percentage}
super_regions={self.super_regions}
dataset={self.dataset}
seed={self.seed}
"""
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _cache_paths(self, key: str) -> Tuple[Path, Path, Path, Path]:
        assert self.index_cache_dir is not None
        self.index_cache_dir.mkdir(parents=True, exist_ok=True)
        meta_pkl = self.index_cache_dir / f"{key}_meta.pkl"
        file_ids_npy = self.index_cache_dir / f"{key}_file_ids.npy"
        rec_ids_npy = self.index_cache_dir / f"{key}_rec_ids.npy"
        trials_pkl = self.index_cache_dir / f"{key}_trials.pkl"
        return meta_pkl, file_ids_npy, rec_ids_npy, trials_pkl

    def _load_or_build_index(self) -> None:
        # Try loading balanced index first (if provided)
        if self._load_balanced_index():
            return

        # Fall back to regular index building
        if not self.session_paths:
            raise ValueError("Either session_paths or balanced_index_path must be provided")

        key = self._index_cache_key()

        if self.index_cache_dir is not None:
            meta_pkl, file_ids_npy, rec_ids_npy, trials_pkl = self._cache_paths(key)
            if meta_pkl.exists() and file_ids_npy.exists() and rec_ids_npy.exists() and trials_pkl.exists():
                if self.verbose:
                    print(f"[SessionDataset] Loading cached index: {key}")

                with open(meta_pkl, "rb") as f:
                    meta = pickle.load(f)
                self.session_path_list = meta["session_path_list"]
                self.region_dict = meta["region_dict"]
                self.chance_accuracy = meta["chance_accuracy"]

                mmap_mode = "r" if self.mmap_index else None
                self.file_ids = np.load(file_ids_npy, mmap_mode=mmap_mode)
                self.rec_ids = np.load(rec_ids_npy, mmap_mode=mmap_mode)

                with open(trials_pkl, "rb") as f:
                    self.trials = pickle.load(f)
                return

        self.region_dict, self.trials = self._build_index_memmap()

        total = sum(self.region_dict.values()) if self.region_dict else 0
        most_common = max(self.region_dict.values()) if self.region_dict else 0
        self.chance_accuracy = (most_common / total) if total > 0 else 0.0

        if self.index_cache_dir is not None:
            meta_pkl, file_ids_npy, rec_ids_npy, trials_pkl = self._cache_paths(key)
            tmp_meta = meta_pkl.with_suffix(".pkl.tmp")
            tmp_trials = trials_pkl.with_suffix(".pkl.tmp")
            tmp_file = file_ids_npy.with_suffix(".tmp.npy")
            tmp_rec = rec_ids_npy.with_suffix(".tmp.npy")

            meta = {
                "session_path_list": self.session_path_list,
                "region_dict": self.region_dict,
                "chance_accuracy": self.chance_accuracy,
            }

            if self.verbose:
                print(f"[SessionDataset] Saving index cache: {key}")

            with open(tmp_meta, "wb") as f:
                pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_meta, meta_pkl)

            np.save(tmp_file, np.asarray(self.file_ids, dtype=np.int32))
            os.replace(tmp_file, file_ids_npy)

            np.save(tmp_rec, np.asarray(self.rec_ids, dtype=np.int32))
            os.replace(tmp_rec, rec_ids_npy)

            with open(tmp_trials, "wb") as f:
                pickle.dump(self.trials, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_trials, trials_pkl)

            if self.mmap_index:
                self.file_ids = np.load(file_ids_npy, mmap_mode="r")
                self.rec_ids = np.load(rec_ids_npy, mmap_mode="r")

    def _build_index_memmap(self) -> Tuple[Dict[str, int], List[str]]:
        region_count_dict: Dict[str, int] = {}
        trials: List[str] = []

        self.session_path_list = []
        path_to_id: Dict[str, int] = {}
        file_ids: List[int] = []
        rec_ids: List[int] = []
        label_ids: List[int] = []  # Store label for each sample (for balanced sampling)

        if self.verbose:
            print("Building recording index...")

        for session_path in self.session_paths:
            sp = str(session_path)
            if sp not in path_to_id:
                path_to_id[sp] = len(self.session_path_list)
                self.session_path_list.append(sp)
            fid = path_to_id[sp]

            with open(sp, "rb") as f:
                session_data = pickle.load(f)

            for i, recording_tuple in enumerate(session_data):
                if not self._keep_recording(sp, i):
                    continue

                meta_str = recording_tuple[1]
                parts = meta_str.split("_")
                if len(parts) < 2:
                    continue

                trials.append(parts[1])

                label_str = parts[-1]
                if self.super_regions:
                    coarse = self._fine_to_coarse_label_map.get(label_str, None)
                    if coarse is None:
                        continue
                    label_key = coarse
                    label_int = self._coarse_label_map.get(coarse, 0)
                else:
                    if label_str not in self._fine_label_map:
                        continue
                    label_key = label_str
                    label_int = self._fine_label_map.get(label_str, 0)

                file_ids.append(fid)
                rec_ids.append(i)
                label_ids.append(label_int)
                region_count_dict[label_key] = region_count_dict.get(label_key, 0) + 1

        self.file_ids = np.asarray(file_ids, dtype=np.int32)
        self.rec_ids = np.asarray(rec_ids, dtype=np.int32)
        self.label_ids = np.asarray(label_ids, dtype=np.int32)  # Store labels

        if self.verbose:
            print(f"Index built. Found {len(self.file_ids)} total recordings.")

        return region_count_dict, trials

    # -------------------------------
    # Dataset interface
    # -------------------------------
    def __len__(self) -> int:
        return int(len(self.file_ids))

    def _load_session(self, session_path: str):
        """
        Load a session pickle.
    
        IMPORTANT:
        - cache_size controls how many *full unpickled pickles* are retained.
        - On Greene, large pickles + cache_size>1 can blow up RAM quickly.
        """
        # Disable payload caching completely
        if self.cache_size <= 0:
            with open(session_path, "rb") as f:
                return pickle.load(f)
    
        # Standard LRU caching
        if session_path in self.cache:
            self.cache.move_to_end(session_path)
            return self.cache[session_path]
    
        with open(session_path, "rb") as f:
            session_data = pickle.load(f)
    
        # Enforce cap strictly
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
        self.cache[session_path] = session_data
        return session_data


    def __getitem__(self, idx: int):
        fid = int(self.file_ids[idx])
        rec_idx = int(self.rec_ids[idx])
        session_path = self.session_path_list[fid]

        session_data = self._load_session(session_path)
        signal_np, meta_str = session_data[rec_idx]

        # Convert to float32 1D with minimal copying
        x = np.asarray(signal_np, dtype=np.float32)
    
        # Optional CPU resample (only if not already resampled and not doing GPU resample)
        if (not self.already_resampled) and (not self.gpu_resample) and (self.signal_sampling_rate != self.target_sampling_rate):
            from scipy.signal import resample
            num_target = int(round(len(x) * self.target_sampling_rate / self.signal_sampling_rate))
            x = resample(x, num_target).astype(np.float32, copy=False)

        # Ensure contiguous once (avoid unconditional extra copy)
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)
            
        x_t = torch.from_numpy(x)
        if not self.include_labels:
            return x_t, ""

        label_str = meta_str.split("_")[-1]
        if self.super_regions:
            coarse = self._fine_to_coarse_label_map.get(label_str, None)
            label_int = int(self._coarse_label_map[coarse]) if coarse is not None else 0
        else:
            label_int = int(self._fine_label_map.get(label_str, 0))

        return x_t, torch.tensor(label_int, dtype=torch.long)

    def get_sample_weights(self) -> np.ndarray:
        """
        Compute sample weights for balanced sampling across classes.
        Weight = 1 / class_frequency, so rare classes get higher weight.

        Returns:
            np.ndarray of shape (len(dataset),) with weight per sample
        """
        if not hasattr(self, 'label_ids') or self.label_ids is None:
            raise RuntimeError("label_ids not available. Rebuild index with updated code.")

        # Count samples per class
        unique_labels, counts = np.unique(self.label_ids, return_counts=True)
        class_counts = dict(zip(unique_labels, counts))

        # Weight = 1 / count (inverse frequency)
        total_samples = len(self.label_ids)
        num_classes = len(unique_labels)

        # Compute weight for each sample based on its class
        weights = np.zeros(total_samples, dtype=np.float64)
        for label, count in class_counts.items():
            mask = self.label_ids == label
            # Weight so that each class contributes equally in expectation
            weights[mask] = 1.0 / count

        # Normalize so weights sum to num_samples (optional, for stability)
        weights = weights / weights.sum() * total_samples

        return weights

    def get_class_distribution(self) -> Dict[str, int]:
        """Return the distribution of classes in this dataset."""
        return dict(self.region_dict) if self.region_dict else {}
