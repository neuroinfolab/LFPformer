# LFPformer

Learn representations of local field potentials and predict anatomical locations
from LFP recordings. LFPformer supports encoder pretraining, Gaussian linear/MLP
and Transformer decoding, micromanipulator fusion, and channel pooling with
geometric refinement.

Bring your preprocessed recordings and a compatible trained checkpoint, or
train a model using the included scripts and configurations. Raw recordings
and trained weights are supplied separately.

## Install

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[train,dev]'
pytest -q
python examples/synthetic_pipeline.py --out outputs/synthetic
```

The CPU post-processing API needs only `pip install -e .`. Install dataset dependencies separately with `.[ibl]` or `.[allen]`. The Allen SDK can require an older NumPy environment; preprocessing and GPU training can run in separate environments. `requirements-runtime.txt` records the tested training versions rather than claiming compatibility with every SDK combination.

See the [code index](docs/code_index.md) for source locations and `tests/` for executable checks.

Using Codex or another coding agent? Read [AGENTS.md](AGENTS.md) for environment
setup, inference commands, experiment workflows, and validation instructions.

## What is included

| Task | Entry point / guide |
|---|---|
| IBL raw `.cbin` preprocessing, 3-s chunks, compact/memory-mapped data | [Data preparation](docs/data.md) |
| Allen SDK LFP preprocessing, metadata and compact shards | `scripts/preprocess_allen.py`, [Allen guide](docs/allen.md) |
| MaskPred, ChanDistc, ManipDistc, CCFDistc pretraining | [Pretraining](docs/pretraining.md), `configs/pretraining/` |
| Frozen LFP embeddings and train-only whitening | `lfpformer embed`, `scripts/whiten_embeddings.py` |
| Linear and Transformer readouts | [Decoders](docs/decoders.md) |
| Actual Allen Gaussian XYZ MLP / frozen and full fine-tuning | `lfpformer reference train-allen-mlp`, [Allen guide](docs/allen.md) |
| Micromanipulator trajectories, linear fusion, folded Transformer fusion | [Micromanipulator guide](docs/micromanipulator.md) |
| Raw predictions, inverse-variance pooling, IBL Kabsch, Allen line projection | [Inference and evaluation](docs/inference.md) |
| NeuroNexus preprocessing, LOSO, frozen readouts and zero-shot evaluation | [NeuroNexus guide](docs/neuronexus.md) |
| Evaluation protocols and checkpoint formats | [Evaluation protocol](docs/paper_protocol.md), [Checkpoint formats](docs/model_registry.md) |

The supported package is `src/lfpformer/`. The preserved paper implementations live in `src/lfpformer/_reference/` and are invoked through named `lfpformer reference` routes. Experimental backbones unrelated to wav2vec2 are not supported by the registry.

## Inference and evaluation

Run a frozen encoder and exported per-window Gaussian linear/MLP head through
embedding extraction, decoding, channel pooling and Kabsch with:

```bash
python scripts/infer_lfp.py --input windows.npz \
  --config configs/pretraining/wav2vec2_ibl_v2_75k_ssl_crossneg_b64_clean.yaml \
  --encoder checkpoints/encoder.pt --decoder checkpoints/best.pt \
  --whitening checkpoints/whitening.npz --device cuda \
  --output-dir outputs/inference
```

`windows.npz` contains preprocessed `signals` (N × 3750; 3 s at 1250 Hz),
`probe_ids`, physical integer `channel_ids`, and `geometry_xyz` (N × 3;
known electrode geometry in µm, repeated for each window). No anatomical
labels are required. Match preprocessing, signal units and encoder config to
training. Use the decoder's saved training whitening; omit `--whitening` only
if that decoder was trained without it. Physical channel IDs must not be
replaced by row indices after channel filtering.

Outputs are `embeddings.npz`, `window_predictions.npz` (means and standard
deviations in AP/DV/ML µm), and `channel_predictions.npz` (refined coordinates,
channel IDs, counts and window-to-channel mapping). `--method pool` skips
Kabsch; `--method allen-line` selects the historical Allen projection.
The default IBL convention expects geometry `(lateral position, 0, shank depth)`;
`--orientation free` disables the IBL shank-orientation constraint.

Choose `--decoder-kind per-window`, `transformer`, `fusion-linear`, or
`fusion-transformer` for the matching checkpoint. All modes support
`--method ibl-kabsch` for inverse-variance channel pooling followed by
unweighted Kabsch. Fusion inputs include row-aligned `manip_xyz` in AP/DV/ML µm.
See [Inference](docs/inference.md) for complete commands and sequence grouping.

An input prediction NPZ contains `mu` and `sigma` of shape `(windows, 3)`, `probe_ids`, integer `channel_ids`, optional `true_xyz`, and known electrode `geometry_xyz`. Coordinates are **AP, DV, ML in µm**; `sigma` is a standard deviation, not variance or log-variance.

```bash
lfpformer postprocess --input predictions.npz --method pool --output outputs/pooled.npz
lfpformer postprocess --input predictions.npz --method ibl-kabsch --output outputs/ibl_kabsch.npz
lfpformer score --input outputs/ibl_kabsch.npz --output outputs/ibl_metrics.json
lfpformer postprocess --input predictions.npz --method allen-line --output outputs/allen_line.npz
```

Kabsch fits **known electrode geometry to predicted coordinates**; it never fits to true anatomical coordinates. Allen's PCA-line projection is a different protocol. See [the metric definitions](docs/paper_protocol.md) before comparing numbers across figures.

## Minimal embedding extraction

```bash
lfpformer chunk --input preprocessed_lfp.npy --probe-id probe01 --output windows.npz
lfpformer embed --input windows.npz \
  --config configs/pretraining/wav2vec2_ibl_v2_75k_ssl_crossneg_b64_clean.yaml \
  --checkpoint checkpoints/encoder.pt --device cuda --output outputs/embeddings.npz
```

The `.npy` input is **already preprocessed** channel × time data at 1,250 Hz. The command does not silently apply IBL destriping to Allen or NeuroNexus. Embedding extraction applies the pinned backbone's resampling policy and writes float16 embeddings, matching the cache boundary used by the readouts.

## Training and evaluation

### Ground truth and split boundaries

Inference-time Kabsch fits **known on-probe electrode geometry to predicted
coordinates**. Histology coordinates and region labels are not model inputs
or fit targets at inference. An end-to-end regression test checks that adding
arbitrary anatomical ground truth leaves predictions unchanged.

Define subject/session-separated train, validation and test splits for your
dataset before constructing training caches.

Fit embedding whitening and target normalization on training data, select
checkpoints with validation data, and use the test set only for final scoring.
The released folded fusion Transformer follows this policy and checks subject
and row overlap before training. Manipulator inputs must be the logged
micromanipulator trajectories available at inference, never histology-corrected
trajectories. The coordinate-conversion helper cannot establish the provenance
of a user-supplied trajectory JSON.

CCFDistc is explicitly an atlas-supervised pretraining comparator, not a
histology-free objective.

- Separate training, validation and test subjects/sessions before building windows.
- Reuse the normalization and whitening saved with the selected checkpoint.
- Use the same channel/probe cohort, decoder, checkpoint and metric for a controlled comparison.
- Figure 3 uses a mean of per-probe RMSE values; this is not channel-aggregated RMSE.
- CCFDistc uses ground-truth atlas coordinates during pretraining and must be labeled supervised/oracle.
- The historical micromanipulator cache uses a zero tip offset. A corrected 200 µm offset changes inputs and requires retraining; see the dedicated guide.
- Checkpoints and compact pickles are Python-serialized artifacts; load only trusted files.

## Citation and licensing

Use the final LFPformer paper citation when its author list and publication metadata are available. This repository intentionally does not invent a DOI, BibTeX entry, or author list. See [CITATION.md](CITATION.md), [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The code is released under the [MIT License](LICENSE); third-party components keep their own licenses (see THIRD_PARTY_NOTICES.md).

Historical absolute paths in `src/lfpformer/_reference/` were anonymised for the public release (`/workspace/ws1/` ... `/workspace/ws5/` stand for the original per-user cluster workspaces); `LFPFORMER_WORKSPACE` relocates them as before.
