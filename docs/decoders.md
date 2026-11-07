# Embeddings and readouts

Extract embeddings with the same frozen encoder checkpoint and preprocessing as the corresponding run. The sequence cache contains flat window-level `emb`, `true_xyz` and `meta_str` arrays, with physical probe/channel metadata encoded in `meta_str`.

```bash
lfpformer reference extract-sequences -- \
  --init_ckpt checkpoints/encoder.pt \
  --config configs/pretraining/wav2vec2_ibl_v2_75k_ssl_crossneg_b64_clean.yaml \
  --probe_data_path data/probe_trainval --eval_data_dir data/eval_folds \
  --out_dir outputs/cache
python scripts/whiten_embeddings.py --cache-dir outputs/cache --out outputs/cache_whit1
lfpformer reference train-linear -- --cache-dir outputs/cache_whit1 \
  --kinds linear --head gauss --seeds 1 2 3 4 5 6 7 42 --out outputs/linear
```

Caches are named `probe_train_seq.npz`, `probe_val_seq.npz` and `test_fold_0_seq.npz` through `test_fold_4_seq.npz`. The original trainers expect all five test caches. Use trusted NPZ caches when they contain object-typed metadata.

The whitening helper fits the included `emb_whiten_stats` implementation to **training embeddings only**, then transforms all splits. Shrinkage 1.0 reduces to feature centering and a global scale; the output RMS normalization is retained. Caches are float16, and inference must preserve that rounding boundary. Save `whitening.npz`; do not fit a new transform on test embeddings or apply it twice.

The per-channel linear/MLP trainer learns a Gaussian mean and diagonal variance with train-only XYZ mean/std normalization. It uses batch 4096, AdamW at 1e-3, weight decay 1e-4, cosine scheduling, up to 600 epochs and validation-loss patience 60. MLP widths start at 256 and halve with added layers. `--head mse` omits the learned variance branch and exports constant per-axis target standard deviations, so within-channel pooling becomes arithmetic averaging. These MLPs are distinct from the Allen BatchNorm/GELU MLP.

## Probe-sequence Transformer

```bash
lfpformer reference train-transformer -- --cache_dir outputs/cache_whit1 \
  --out_dir outputs/transformer --arch transformer --d_model 256 \
  --layers 4 --nhead 8 --ff 512 --epochs 200 --batch_size 8 --seed 1 \
  --gaussian --max_len 4096
```

This is a runnable direct-XYZ example. Set all arguments to the recorded run protocol to reproduce a particular figure. `--decompose` enables the historical rigid-pose-plus-residual head; omitting it selects direct XYZ. They are different model parameterizations. `configs/decoding/*.yaml` describe recipes for inspection; they are not automatically consumed by the original argparse trainers.

Use `scripts/predict_head.py` for exported per-channel `best.pt` or Allen `best_gaussian_xyz_model.pt` heads. Sequence trainers export their test predictions directly. Their checkpoints and manipulator-conditioned checkpoints require their respective model loaders, not the per-channel predictor.

The packaged `GaussianXYZHead` in `lfpformer.heads` matches the actual Allen MLP state-dict layout. The separate `LinearGaussianHead` is a small utility; the paper capacity-ablation checkpoints are loaded using the preserved `GaussHead` class.
