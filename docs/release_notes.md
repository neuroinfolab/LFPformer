# Package entry points

Use `lfpformer --help` to discover preprocessing, embedding, postprocessing and
scoring commands. Named `lfpformer reference` routes expose training tools.

- `scripts/infer_lfp.py`: encoder and decoder inference with channel pooling
  and optional geometric refinement.
- `scripts/configure_pretraining.py`: resolve dataset/output paths in a
  pretraining configuration.
- `scripts/whiten_embeddings.py`: fit and save an embedding transform.
- `scripts/prepare_manipulator.py`: convert logged insertion metadata into
  row-aligned channel coordinates.
- `scripts/train_fusion.py`: prepare subject splits and train a fusion decoder.

Use explicit paths for your datasets and checkpoints. The environment variable
`LFPFORMER_WORKSPACE` relocates defaults in the bundled training implementations.
Generated data, caches, checkpoints and outputs remain outside version control.
