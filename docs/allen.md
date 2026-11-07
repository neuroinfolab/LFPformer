# Allen preprocessing, MLP and line fitting

The raw Allen entry point reads Visual Coding Neuropixels LFP through the Allen SDK. Run preprocessing in an SDK-compatible environment, then transfer the compact shards to the training environment.

```bash
python scripts/preprocess_allen.py --manifest data/allen/manifest.json \
  --channels-csv data/allen/train_channels.csv --atlas-metadata data/allen/metadata.pkl \
  --out data/allen_compact/train
```

The CSV must contain `session_id,probe_id,channel_id,acronym,AP,DV,ML,label`; it explicitly defines split membership and the fixed atlas labels. Repeat using separate validation/test CSVs. The script requires 1,250-Hz SDK LFP, keeps finite full 3-second windows and applies no additional filtering. The original `preprocessing/allen/spectrogram_data_creator.py` is also included; despite its name its relevant waveform branch writes raw LFP chunks. Exact historical row ordering and time exclusions must be preserved when reusing published checkpoint predictions.

The fixed metadata/atlas mapping and split lists remain required inputs. The convenience exporter does not claim that arbitrary new CSV splits reproduce the paper. Its complete-window endpoint policy is documented in the exported `preprocessing.json`.

## Actual XYZ MLP

```bash
lfpformer reference train-allen-mlp -- \
  --config configs/pretraining/wav2vec2_probe_dist_v2_embedonly_full.yaml \
  --checkpoint checkpoints/encoder.pt --data-dir data/allen_eval \
  --output-dir outputs/allen_frozen --lr-backbone 0 --lr-cnn 0 \
  --epochs 50 --batch-size 32 --lr-head 0.001
```

The Gaussian head is `D → 512 → 256 → 128`, with BatchNorm, GELU and dropout 0.2/0.2/0.1, then separate three-coordinate mean/log-variance outputs. Log variance is clamped to [-10,10]. It uses training XYZ mean/std and Gaussian NLL. The default warmup is three epochs and validation patience is ten. For full fine-tuning use `--lr-backbone 5e-6 --lr-cnn 5e-6` and preserve the recorded configuration. The trainer exports the best backbone and head states, target normalization and predictions.

`extract-allen-sequences` preserves the sequence-cache extraction path. Zero-shot IBL→Allen transfer, frozen Allen MLP fitting, full Allen fine-tuning and native Allen pretraining are separate conditions. In particular, the historical ChanDistc and ManipDistc zero-shot heads can have different rigid/decomposed versus direct-XYZ parameterizations.

## Paper post-processing

The Allen paper pipeline pools repeated windows by physical channel with per-axis `1/(sigma²+1e-6)` weights, fits an unweighted PCA line per probe and projects the pooled predictions onto that line. It then broadcasts the projected channel predictions back to windows for the historical five-fold region scores. Use the output `window_to_channel` index to perform that broadcast.

The fixed evaluation universe has 116 regions. Atlas lookup floors 25-µm voxel coordinates in AP/DV/ML order and treats out-of-atlas predictions as wrong. The included `fig4cr_allen_transfer_compute.py` and `compute_allen_per_fold_balacc.py` preserve the figure-specific scoring. IBL Kabsch is not the Allen PCA-line procedure, and coordinate RMSE is not balanced classification accuracy.
