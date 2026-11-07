# Inference and evaluation

The portable pipeline is preprocessed LFP → frozen embeddings → decoder predictions → dataset-specific post-processing → scoring. The exact checkpoint/config and any training-fitted normalization are required.

```bash
lfpformer chunk --input preprocessed_lfp.npy --probe-id probe01 --output outputs/windows.npz
lfpformer embed --input outputs/windows.npz \
  --config configs/pretraining/wav2vec2_ibl_v2_75k_ssl_crossneg_b64_clean.yaml \
  --checkpoint checkpoints/encoder.pt --output outputs/embeddings.npz --device cuda
python scripts/predict_head.py --input outputs/embeddings.npz \
  --checkpoint checkpoints/linear_s1/best.pt --whitening checkpoints/whitening.npz \
  --output outputs/predictions.npz --device cuda
```

For an Allen fine-tuned model, use its exported backbone for embedding extraction and its matching `head_state_dict` checkpoint for prediction. Do not use the original frozen encoder after fine-tuning the backbone. Omit whitening for heads trained without it.

## Prediction schema

| Key | Shape | Meaning |
|---|---|---|
| `mu` | N × 3 | Predicted AP/DV/ML in µm |
| `sigma` | N × 3 | Positive per-axis standard deviation in µm |
| `probe_ids` | N | Stable probe identifier, Unicode string array |
| `channel_ids` | N | Nonnegative integer physical channel index |
| `true_xyz` | N × 3 | Optional evaluation coordinates, never used for fitting |
| `geometry_xyz` | N × 3 | Optional known electrode geometry, required by IBL Kabsch |
| `true_labels` | N | Optional fixed atlas-class labels |

`pred_xyz_gauss`/`pred_sigma_gauss` are accepted aliases. Use `python scripts/convert_predictions.py --input historical.npz --output predictions.npz` to convert trusted IBL `meta_str` into explicit probe/channel IDs; the original parser is in `recompute_refined_3d_metrics.py`. `lfpformer.geometry.np1_geometry(channel_ids)` supplies the paper NP1 staggered geometry; it is not a geometry definition for every probe type.

## Raw, pooled and fitted outputs

```bash
lfpformer postprocess --input predictions.npz --method raw --output outputs/raw.npz
lfpformer postprocess --input predictions.npz --method pool --output outputs/pool.npz
lfpformer postprocess --input predictions.npz --method ibl-kabsch --output outputs/kabsch.npz
lfpformer postprocess --input predictions.npz --method allen-line --output outputs/line.npz
lfpformer score --input outputs/kabsch.npz --output outputs/metrics.json
```

Pooling groups `(probe_id, channel_id)` and weights each coordinate axis separately by inverse variance. IBL uses `1/max(sigma²,1e-6)`; historical Allen uses `1/(sigma²+1e-6)`. If uncertainty is absent, the operation uses a simple mean. Fitted outputs deliberately do not report a newly calibrated sigma: the original windows' uncertainties are not an uncertainty model for the fitted trajectory.

IBL Kabsch fits a rigid transform from **known electrode geometry to pooled predictions**, with equal weight per channel. The paper's orientation constraint flips the fitted shaft if its DV component is positive, then recomputes translation. Probes with fewer than four channels retain their predictions. `--orientation free` disables that paper-specific constraint; it is a different evaluation setting.

Allen line fitting performs an unweighted PCA on pooled predicted channel coordinates and projects them onto the first principal axis. It neither uses true coordinates nor enforces channel spacing. Probes with fewer than three channels retain their predictions. `window_to_channel` in the output maps the original windows to pooled channels; use it to broadcast predictions back to windows for historical Allen fold-level scoring.

Scoring folds only ML about 5,739 µm (`ML := 5739 - abs(ML-5739)`) by default. It reports both channel-aggregated 3D RMSE and the mean/sample-SD of per-probe RMSE. Supply `--no-fold` only for a native-hemisphere analysis. Score pooled outputs to obtain a physical-channel metric; scoring raw windows instead gives a window-weighted metric even though the generic row count field is called `n_channels`.

## Atlas labels and balanced accuracy

`lfpformer atlas-labels --input outputs/line.npz --volume atlas_class_ids.npy --output outputs/labels.npz` requires a volume already in **AP × DV × ML order**, with voxel values matching the fixed class IDs. The default voxel size is 25 µm. Out-of-volume predictions receive label -1; do not clip them to the nearest edge. Atlas background values must likewise be mapped to the intended non-evaluable class before scoring.

`lfpformer balanced-accuracy --input outputs/labels.npz --classes class_ids.json --output outputs/balacc.json` computes mean recall over true classes present in the supplied fixed class universe. Unknown true labels are rejected; predictions outside the universe, including -1, count as wrong. Do not silently drop failed atlas lookups. Retain the fixed 309-region IBL or 116-region Allen mapping and the original fold aggregation when reproducing Figure 4.

## One-command inference for all released decoder families

`infer_lfp.py` accepts four `--decoder-kind` values. Each writes embeddings,
window-level Gaussian predictions, and channel-level outputs. All commands
below explicitly request inverse-variance pooling followed by unweighted Kabsch.
Run from the repository root after installing `.[train]`.

### LFP-only linear or MLP

```bash
python scripts/infer_lfp.py --input data/windows.npz \
  --config configs/pretraining/wav2vec2_ibl_v2_75k_ssl_crossneg_b64_clean.yaml \
  --encoder checkpoints/encoder.pt --decoder checkpoints/linear_best.pt \
  --decoder-kind per-window --whitening checkpoints/whitening.npz \
  --method ibl-kabsch --device cuda --output-dir outputs/linear_inference
```

### LFP-only Gaussian Transformer

```bash
python scripts/infer_lfp.py --input data/windows.npz \
  --config configs/pretraining/wav2vec2_ibl_v2_75k_ssl_crossneg_b64_clean.yaml \
  --encoder checkpoints/encoder.pt --decoder checkpoints/transformer_best.pt \
  --decoder-kind transformer --method ibl-kabsch \
  --device cuda --output-dir outputs/transformer_inference
```

This loads the `ProbeSequenceModel` architecture from checkpoint `args`, with
strict state-dict loading, including direct-XYZ or decomposed-pose variants.
Saved `emb_mean`/`emb_std`/`emb_W` are applied automatically. If training instead
used an externally whitened cache, supply its saved `--whitening` file. The
loader rejects simultaneous checkpoint normalization and external whitening.

### LFP + micromanipulator Gaussian linear/MLP

```bash
python scripts/infer_lfp.py --input data/windows_with_manip.npz \
  --config configs/pretraining/wav2vec2_ibl_v2_75k_ssl_manip_dist_fresh.yaml \
  --encoder checkpoints/finetune_best_valmse.pt \
  --decoder checkpoints/finetune_best_valmse.pt \
  --decoder-kind fusion-linear --method ibl-kabsch \
  --device cuda --output-dir outputs/linear_fusion_inference
```

Use the backbone bundled in the fusion checkpoint, including when it was
fine-tuned. The loader uses `gauss_head_state_dict`, `normalizer_stats`, and
the saved direct/residual parameterization. It preserves float32 embeddings,
matching the original online linear-fusion evaluation. Do not supply whitening.

### LFP + micromanipulator folded Gaussian Transformer

```bash
python scripts/infer_lfp.py --input data/windows_with_manip.npz \
  --config configs/pretraining/wav2vec2_ibl_v2_75k_ssl_manip_dist_fresh.yaml \
  --encoder checkpoints/encoder.pt --decoder checkpoints/fusion_best.pt \
  --decoder-kind fusion-transformer --method ibl-kabsch \
  --device cuda --output-dir outputs/transformer_fusion_inference
```

This supports `MatchedDirectProbeSeq` checkpoints produced by `train_fusion.py`.
It restores saved embedding centering/scaling, folds manipulation coordinates
using the original model implementation, and respects the checkpoint's
`lfp_manip` or `lfp_only` arm. No external whitening is applied.

### Inputs and sequence context

All modes accept `signals`, `probe_ids`, `channel_ids`, and `geometry_xyz` as
described above. Fusion adds `manip_xyz`, a finite `(N,3)` array of logged
micromanipulator coordinates in AP/DV/ML micrometres, aligned to signal rows.
The `lfp_only` fusion control does not need this array. None of these modes
requires `true_xyz`, region labels or a histology trajectory.

Transformer geometry columns must be `(x, 0, depth)` in micrometres. By default,
each probe forms one attention sequence, sorted by depth. Optional Unicode
`sequence_ids` assigns windows to explicit attention contexts, each containing
only one physical probe. Predictions are restored to original row order before
pooling across all windows for each physical channel. Sequences longer than
the checkpoint's `max_len` cause an error instead of silently subsampling.
Choose sequence membership to match the intended evaluation protocol; changing
it changes the context available to the Transformer. `--batch-size` controls
encoder extraction, while sequence decoding runs one sequence at a time.

The selected encoder config must match the encoder checkpoint in every example.
Only the documented checkpoint formats are accepted; incompatible weights fail
strict loading. Use `--method pool` to retain pooled predictions without fitting
or `--method allen-line` for the Allen projection. Kabsch never uses anatomical
ground truth. Result coordinates are in AP/DV/ML micrometres; hemisphere folding
for scoring follows the separate `lfpformer score` options.
