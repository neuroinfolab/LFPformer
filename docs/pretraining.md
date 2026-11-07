# Pretraining

The release pins the actual wav2vec2 implementations used in the training and evaluation stacks. Configurations retain the original model, optimizer and loss parameters; local paths and worker counts have been made portable.

| Objective | Scaling configuration | Training route |
|---|---|---|
| MaskPred, Base | `scaling_base_maskpred_lr1p02e-4.yaml` | `pretrain` |
| ChanDistc, Base | `scaling_base_chandistc_lr1p02e-4.yaml` | `pretrain` (channel-aware step sampler) |
| ManipDistc, Base | `ab_manip_lr1p02e-4.yaml` | `pretrain-manipdistc` |
| CCFDistc, Base | `ab_ccf_lr1p02e-4.yaml` | `pretrain` |
| MaskPred/ChanDistc, Large | `scaling_large_*_lr1p5e-4.yaml` | `pretrain` |
| ManipDistc/CCFDistc, Large | `gate_large_*_lr1p5e-4.yaml` | respective route above |

CCFDistc uses ground-truth atlas coordinates during pretraining and is an oracle/supervised comparator. ManipDistc replaces these with manipulation-derived coordinates. ChanDistc uses within-probe channel geometry. The scaled ChanDistc sampler groups samples from the same probe; an ordinary random sampler is not equivalent.

## Resolve paths and run

```bash
python scripts/configure_pretraining.py \
  --config configs/pretraining/scaling_base_chandistc_lr1p02e-4.yaml \
  --data data/ibl_fullrec --validation data/ibl_fullrec/val_subset \
  --results outputs/chandistc --max-steps 35000 --output outputs/chandistc/run.yaml
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m lfpformer.cli reference pretrain -- --backbone wav2vec2 \
  --config outputs/chandistc/run.yaml
```

Supply `--probe-data` to the configuration helper if running the original intermediate supervised linear-probe monitor. Without it the helper disables that monitor; this does not change the SSL loss. The training config requires a real `val_subset_dir`; it cannot be omitted in step mode. Preserve validation membership and cadence for checkpoint comparisons.

The scaling YAMLs specify **global** `batch_size_ssl=512`, divided across ranks. `GRAD_ACCUM_STEPS` multiplies the effective sample count per optimizer update; leave it unset/1 for the documented 512-example schedule. At batch 512, 35,000 updates represent 17.92 million sample presentations; 115,000 represent 58.88 million. A step count is not itself a dataset size. Hold effective batch, sampler, learning-rate schedule and validation rule fixed when changing GPU count. Avoid assuming that stopping a 120k-step schedule at 35k is equivalent to training a newly shortened 35k cosine schedule.

For ManipDistc, generate **row-aligned** `train_manip_xyz.npy`, `train_valid.npy`, `val_manip_xyz.npy`, `val_valid.npy` and matching arrays for any validation subset used by the loader. Set `MANIP_XYZ_DIR` to that directory and use `reference pretrain-manipdistc`. Missing/degenerate trajectories are excluded. The exact array lengths and order must match the compact or memory-mapped split. `build_manip_memmap.py` and `precompute_manip_xyz_compact.py` are included for the two representations.

The `pretrain-chandistc` route preserves the older epoch-based grouped-probe trainer. It has explicit CLI distance-loss settings, including sigma and adaptive weight, and is distinct from the scaling step-mode recipe. Do not use its CLI defaults as substitutes for the scaling YAML.

The `75k` filename tag refers to the historical dataset-scale family, **not 75,000 optimizer updates**. Epoch 53 / epoch 164 checkpoint identifiers belong to that older recipe. The Base/Large scaling runs use a different full-recording recipe; matching encoder name alone does not make them controlled comparisons.

## Resume and initialization

The training entry point accepts `--resume_from`; inspect `--help` for the explicit LR-override options. Preserve optimizer/scheduler state for a true continuation. `rand_init: false` loads the named audio-pretrained wav2vec2 initialization. Step zero can therefore have useful structure before LFP pretraining. An LFP checkpoint and its matching config must be supplied to inference; random initialization is never silently substituted for missing weights.
