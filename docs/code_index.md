# Code index

| Location | Contents |
|---|---|
| `src/lfpformer/geometry.py` | ML folding, electrode geometry, rigid Kabsch and PCA-line projection |
| `src/lfpformer/pooling.py` | Physical-channel pooling with per-axis uncertainty weights |
| `src/lfpformer/metrics.py`, `atlas.py` | Probe/channel RMSE, seed aggregation, atlas labels and balanced accuracy |
| `src/lfpformer/inference.py`, `heads.py` | Pinned-backbone embedding extraction and Gaussian heads |
| `scripts/` | Portable config, Allen preprocessing, manipulation, whitening, inference and fusion wrappers |
| `_reference/pretraining/` | Original pretraining loss/backbone stack, DDP trainers and scaling samplers |
| `_reference/benchmark/` | Original decoder, embedding, transfer and figure-evaluation implementations |
| `_reference/blind_localization/` | Dataset readers, waveform handling and original model utilities |
| `_reference/preprocessing/ibl/` | ONE metadata helpers, raw blocks, compact/memory-map construction and splits |
| `_reference/preprocessing/allen/` | Historical Allen waveform and atlas-dataset preparation |
| `_reference/preprocessing/neuronexus/` | Historical NeuroNexus acquisition preprocessing |
| `_reference/fusion/` | Published folded direct-XYZ Transformer-fusion experiment and model |
| `configs/` | Pretraining templates and decoder recipe descriptions |

`_reference` paths above are relative to `src/lfpformer/`. Use `lfpformer reference NAME -- --help` to select a packaged command without reproducing the original machine's `PYTHONPATH` layout.
