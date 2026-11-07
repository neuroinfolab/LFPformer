# Micromanipulator coordinates and fusion

Manipulation trajectories are anatomical inputs in AP/DV/ML µm. For left-hemisphere-folded training, fold **both** the manipulator input and the XYZ target on the ML axis. Folding only the prediction at evaluation leaves the model input in a different coordinate convention.

## Build coordinates

```bash
python scripts/prepare_manipulator.py --rows windows.npz \
  --trajectories trajectories.json --tip-offset-um 0 --out data/manip_xyz.npy
```

The row file has `probe_ids` in `EID_probeXX` format and physical `channel_ids`. Each trajectory JSON record has `eid`, `probe`, `x`, `y`, `z`, `depth`, `theta`, `phi`, as expected by IBL's `Insertion.from_dict`. The script uses the IBL 25-µm atlas coordinate conversion. Missing or zero-depth trajectories produce invalid rows; preserve a validity mask and exclude those rows consistently across comparison arms.

The historical paper cache uses a **0 µm tip offset**, with row depth `(channel//2)*20`. A physically corrected 200 µm tip offset changes model inputs and must be evaluated with retrained models. It cannot be substituted only at test time. `precompute_manip_xyz.py`, `precompute_manip_xyz_compact.py` and `build_manip_memmap.py` preserve the original per-split workflows and filename conventions for downstream and pretraining datasets.

## Linear fusion

```bash
lfpformer reference train-manip-linear -- --mode supervised \
  --init_ckpt checkpoints/manipdistc_ep53.pt \
  --config configs/pretraining/wav2vec2_ibl_v2_75k_ssl_manip_dist_fresh.yaml \
  --probe_data_path data/probe_trainval --manip_xyz_dir data/manip_xyz \
  --results_dir outputs/linear_manip --freeze_backbone --head_type linear \
  --parameterization direct --subject_val_frac 0.2 \
  --subject_map data/session_to_subject.json --split_seed 0 --seed 1
lfpformer reference eval-manip-linear -- \
  --checkpoint outputs/linear_manip/checkpoints/finetune_best_valmse.pt \
  --eval_data_dir data/eval_folds --manip_xyz_dir data/manip_xyz_eval \
  --output_dir outputs/linear_manip/evaluation
```

The historical recipe uses 20 epochs, batch 64 windows, LR 1e-4, 10% warmup then cosine to 1% of peak, weight decay 1e-4 and early-stop patience 8. Its target/manip normalization uses AP/DV/absolute-ML ranges. Model selection uses the MSE branch's validation loss while also exporting the Gaussian branch. Preserve these details for reproduction, and change them explicitly for a new controlled comparison.

## Folded Transformer fusion

`scripts/train_fusion.py` wraps the actual folded direct-XYZ implementation. Run `prepare` once, then `train` for each seed:

```bash
python scripts/train_fusion.py prepare --out outputs/fusion \
  --cache-dir data/manipdistc_cache --manip-dir data/manip_xyz_eval \
  --subject-map data/trainval_session_to_subject.json \
  --all-subject-map data/all_session_to_subject.json --encoder manipdistc_ep53
python scripts/train_fusion.py train --out outputs/fusion \
  --cache-dir data/manipdistc_cache --manip-dir data/manip_xyz_eval \
  --subject-map data/trainval_session_to_subject.json \
  --all-subject-map data/all_session_to_subject.json --encoder manipdistc_ep53 \
  --arm lfp_manip --seed 1
```

Repeat for seeds 1–7 and 42. `lfp_only` zeros the manipulator inputs while retaining the same input projection. The encoder argument records provenance; embeddings must already have been generated using that encoder. Preparation validates subject separation, metadata alignment, and normalization for the supplied cohort.

The Transformer recipe is four layers, width 256, eight heads, feed-forward 512, sequence limit 4096, batch 8 sequences, 200 epochs, LR 3e-4 constant, weight decay 1e-4. It uses a subject-separated training/validation split and training-subject embedding centering/global-RMS scaling. Folded manipulation inputs and targets use the same frame. Preparation writes the split and normalization into the output directory.
