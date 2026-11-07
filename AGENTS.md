# Agent guide for LFPformer

These instructions apply throughout this repository. They are intended for
Codex and other coding agents that support AGENTS.md. Run commands from the
repository root. Follow the user's requested experiment scope and use the
existing implementations as the starting point. This is a toolkit for users to
run on their own data; write documentation as setup and usage instructions.

## Start here

1. Read `README.md` and `docs/code_index.md`.
2. Inspect `git status --short` and preserve existing user changes.
3. Read the guide for the task: `docs/pretraining.md`, `docs/decoders.md`,
   `docs/micromanipulator.md`, or `docs/inference.md`.
4. Identify the dataset paths, encoder config/checkpoint, decoder checkpoint,
   normalization artifacts, split manifest and available compute before a run.
5. Put generated artifacts in `outputs/<experiment-name>/` and retain the
   resolved configuration, commands, logs, checkpoint selection and metrics.

## Environment setup

Python 3.10 or newer is required. Reuse a working environment when provided.
For a fresh installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train,dev]'
python -m pytest -q
python examples/synthetic_pipeline.py --out outputs/synthetic
```

Install mutually compatible PyTorch and torchaudio builds for the available
CPU/CUDA environment. `requirements-runtime.txt` records training versions.
Optional dataset dependencies are `.[ibl]`, `.[allen]`, and `.[neuronexus]`;
use separate preprocessing environments when SDK dependencies conflict.

`pip install -e '.[dev]'` is sufficient for NumPy-based postprocessing tests;
model tests skip without training dependencies. Report skips as well as passes.
Use `torch.cuda.is_available()` to verify GPU access before selecting CUDA.

## Repository map

- `src/lfpformer/`: supported pooling, geometry, metrics, inference and CLI.
- `scripts/`: portable experiment and inference wrappers.
- `src/lfpformer/_reference/`: preserved training and evaluation implementations.
- `configs/pretraining/`: model, objective and optimizer configurations.
- `configs/decoding/`: human-readable decoder recipes; original trainers take
  CLI arguments rather than loading these YAMLs automatically.
- `tests/`: numerical equivalence, model, fusion and split tests.

Discover entry points with `lfpformer --help` and
`lfpformer reference <route> -- --help`. Use the named routes instead of
reconstructing cluster-specific Python import paths. Run separate processes
for different reference stacks. `LFPFORMER_WORKSPACE` relocates historical
defaults; explicit CLI paths are preferable.

## Data and checkpoint contract

Raw recordings and trained model weights are supplied separately. Obtain paths
from the user or the documented storage location; do not substitute synthetic
data or random weights for a requested scientific experiment.

Window inputs are preprocessed single-channel LFP, 3 seconds at 1250 Hz:
`signals` has shape `(N, 3750)`. Preserve the training preprocessing and units.
NPZ metadata uses Unicode `probe_ids`, integer physical `channel_ids`, and,
for Kabsch, `geometry_xyz` with shape `(N, 3)` in micrometres. Repeat physical
geometry for repeated windows of the same channel. Preserve physical channel
IDs after filtering. NP1 geometry is available through
`lfpformer.geometry.np1_geometry`; other probes need their own geometry.

Prediction coordinates are `(AP, DV, ML)` in micrometres. `sigma` is a positive
standard deviation, not a variance or log-variance. Load the encoder with its
matching config and reuse the decoder's saved normalization/whitening.

## Inference

For an exported per-window Gaussian linear/MLP decoder:

```bash
python scripts/infer_lfp.py --input data/windows.npz \
  --config configs/pretraining/wav2vec2_ibl_v2_75k_ssl_crossneg_b64_clean.yaml \
  --encoder checkpoints/encoder.pt --decoder checkpoints/best.pt \
  --whitening checkpoints/whitening.npz --device cuda \
  --output-dir outputs/inference
```

Select the actual encoder config for the checkpoint. Omit whitening only when
the decoder was trained without it. This writes embeddings, window predictions
and refined channel predictions. Anatomical ground truth is not required.
Use `--method pool` for pooling alone or `--method allen-line` for the historical
Allen projection. Use `--decoder-kind transformer` for LFP-only Gaussian sequence checkpoints,
`fusion-linear` for the released manipulation-conditioned linear/MLP checkpoints,
or `fusion-transformer` for folded MatchedDirectProbeSeq checkpoints.
All modes run inverse-variance channel pooling and unweighted Kabsch by default;
make this explicit with `--method ibl-kabsch`. See `docs/inference.md` for
checkpoint normalization, sequence grouping and manipulation input requirements.

## Run experiments

For pretraining, resolve data and output paths into a run-specific YAML:

```bash
python scripts/configure_pretraining.py \
  --config configs/pretraining/scaling_base_chandistc_lr1p02e-4.yaml \
  --data data/ibl_fullrec --validation data/ibl_fullrec/val_subset \
  --results outputs/chandistc --output outputs/chandistc/run.yaml
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m lfpformer.cli reference pretrain -- --backbone wav2vec2 \
  --config outputs/chandistc/run.yaml
```

Choose GPU count and run budget for the available hardware. Scaling configs
specify a global batch size; preserve effective batch size when changing ranks
or gradient accumulation. Changing the total step count can change the learning
rate schedule. Run a small, separately named smoke experiment before a full run.
ManipDistc uses `pretrain-manipdistc` and row-aligned manipulation arrays through
`MANIP_XYZ_DIR`; follow `docs/pretraining.md` for the required filenames.

For frozen decoding: extract embeddings, apply the saved/train-fitted whitening,
then use `train-linear` or `train-transformer` as documented in `docs/decoders.md`.
For fusion: prepare logged micromanipulator coordinates, then follow
`docs/micromanipulator.md` for `train-manip-linear` or the `train_fusion.py`
prepare/train workflow. Preserve the selected recipe and seed list.

Record the command, Git commit, seed, input/checkpoint identifiers, split,
resolved config, environment, selected epoch and selection metric for each run.
Use validation for model selection and reserve test scoring for final evaluation.
Keep subjects/sessions/probes separated according to the dataset protocol.
Use logged micromanipulator trajectories as inputs, not histology trajectories.

## Geometry and evaluation rules

- Pool repeated windows by `(probe_id, channel_id)`, separately for each axis.
- IBL weights are `1 / max(sigma**2, 1e-6)`; its Kabsch fit is unweighted.
- Kabsch fits known electrode geometry to pooled predictions. Never fit to
  `true_xyz` or use anatomical test labels to construct electrode geometry.
- Allen PCA-line projection and IBL Kabsch are different protocols.
- Preserve the requested hemisphere convention, atlas mapping and cohort.
- Distinguish window-weighted, channel-aggregated and equal-probe metrics.
- Keep decoder architecture, postprocessing and evaluation cohort matched when
  the experiment is intended to compare encoder objectives.
- Identify CCFDistc as atlas-supervised when describing its results.

## Validate and deliver

Run relevant tests after changes; for model/inference changes use the training
environment and `python -m pytest -q`. Use a small synthetic end-to-end run to
check mechanics before running on supplied recordings. Never report synthetic
metrics as paper results. Do not change recorded paper numbers without an
evaluation artifact supporting the change.

Keep data, checkpoints, credentials, environment files and large generated
outputs out of Git. Before an authorized push, inspect the diff and staged
files, run `git diff --check`, and confirm the destination remote. Report what
changed, the tests run, output locations and whether the push succeeded.
