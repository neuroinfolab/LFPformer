# Evaluation protocols

Choose an evaluation protocol for your dataset and apply it consistently across
models. Record the split, channel/window cohort, decoder, normalization and
aggregation method with the output metrics.

| Task | Processing | Scoring |
|---|---|---|
| Probe localization | Inverse-variance channel pooling, known-geometry Kabsch | Channel-aggregated RMSE or equal-probe mean RMSE |
| Allen line projection | Channel pooling, PCA-line projection | Localization metrics on the selected cohort |
| Anatomical classification | Map predicted coordinates to a fixed atlas class volume | Balanced accuracy and confusion matrices |
| Direct region classification | Classifier or reference-bank predictions | Balanced accuracy on held-out sessions/subjects |
| Pretraining scaling | Fixed decoder and evaluation protocol at selected checkpoints | Report optimizer updates, effective batch size and sample presentations |

Kabsch uses electrode geometry and predicted coordinates only. Ground-truth
coordinates are needed for localization scoring, not for inference. Channel
pooling uses physical `(probe_id, channel_id)` pairs. Do not merge channels from
different insertions or treat repeated windows as separate electrodes.

Specify whether scoring uses native coordinates or ML folding and retain that
choice across comparisons. Atlas volumes must use the expected AP/DV/ML axis
order, voxel size and class mapping. Count failed predicted atlas lookups as
incorrect assignments rather than silently dropping them.

Channel-aggregated RMSE and equal-probe mean RMSE weight observations differently.
For repeated decoder seeds, specify whether seeds are averaged before or after
probe aggregation. State whether standard deviations describe seeds, probes,
subjects or folds.

Localization error and classification accuracy measure different outcomes:
a nearby coordinate can cross an atlas boundary. Report both when useful.
Use matched decoders, preprocessing and data budgets when comparing encoder
objectives. CCFDistc uses atlas coordinates in pretraining and is an explicitly
supervised comparator.

See [Inference](inference.md) for executable scoring commands and
[Data preparation](data.md) for input formats.
