# NeuroNexus

The release contains raw preprocessing, embedding/prediction, nearest-neighbor transfer, post-hoc refinement and supervised leave-one-session-out (LOSO) evaluation. Source preprocessing is under `src/lfpformer/_reference/preprocessing/neuronexus/`; the raw pipeline preserves its filtering, ADC-unit conversion, trial segmentation and session metadata logic. It needs the recording's `.dat`/`.h5`/`.xml`/label files and its acquisition settings. The raw script has dataset-specific acquisition settings rather than a universal one-command converter.

Install `.[train,neuronexus]` for the raw preprocessing dependencies.

The five labels are Cortex, CA1, CA2, CA3 and DG. The paper-aligned seven-session pool is `AD_HF01_1`, `AD_HF02_2`, `AD_HF02_4`, `AD_HF03_1`, `AD_HF03_2`, `NN_syn_01`, `NN_syn_02`. `NN_syn_03` is excluded because it is Cortex-only.

```bash
lfpformer reference neuronexus-loso -- --variant ibl_cd500_75k \
  --test-session AD_HF01_1 --lfp-dir data/neuronexus/lfp \
  --checkpoint checkpoints/ibl_chandistc.pt --output-base outputs/neuronexus_loso \
  --no-wandb
lfpformer reference neuronexus-unified -- --mode frozen_lp --backbone-type wav2vec2 \
  --ckpt checkpoints/encoder.pt \
  --config configs/pretraining/wav2vec2_ibl_v2_75k_ssl_crossneg_b64_clean.yaml \
  --lfp-dir data/neuronexus/lfp --out-dir outputs/neuronexus_frozen
```

Run LOSO over each held-out session. The supervised implementation uses 60 trials/session with 12 fixed random test trials, and training/validation trials drawn from the remaining 48. It assigns four of the other six sessions to training and two to validation, preserving disjoint trial pools. Its complete implementation, including seed and class order, is included; do not replace this with a random window split.

The unified route exposes `zero_shot`, `frozen_lp` and `full_ft`. Additional preserved routes are `neuronexus-embed`, `neuronexus-knn` and `neuronexus-refine`; each has its original `--help`. Only the wav2vec2 backbone is supported by the release registry. References to unrelated rival architectures are historical branches, not packaged model support.

Region labels do not establish full XYZ ground truth or hemisphere. Do not interpret a region-only NeuroNexus evaluation as an IBL-style coordinate-RMSE result. Some historical zero-shot records contain hard best-k labels only; these do not support a newly calibrated probability or ROC analysis. State whether reported scores are hard-label accuracy, balanced accuracy or probability-based metrics.
