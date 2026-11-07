# Checkpoint formats

Use an encoder and decoder trained together, with their matching configuration
and normalization. Checkpoints are supplied by the user or produced by training.

| Model | Required checkpoint contents | Inference mode |
|---|---|---|
| wav2vec2 encoder | `backbone_state_dict` and matching encoder YAML | All modes |
| Per-window Gaussian linear/MLP | `model`, `kind`, `mu_t`, `sd_t`; optional `head`, `mlp_layers` | `per-window` |
| Allen Gaussian MLP | `head_state_dict`, `y_mean`, `y_std`; matching backbone | `per-window` |
| Gaussian probe Transformer | `model`, architecture `args`; saved embedding normalization when fitted internally | `transformer` |
| Micromanipulator linear/MLP | `gauss_head_state_dict`, `normalizer_stats`, `head_type`, `parameterization`, `manip_conditioned`; matching bundled backbone | `fusion-linear` |
| Folded fusion Transformer | `model` with `embedding_mean` and `embedding_scale`, architecture `args` and `arm` | `fusion-transformer` |

The inference loader checks parameter names and shapes strictly. Supply external
whitening only for models trained on externally whitened caches. Do not apply
it again when the checkpoint already includes embedding normalization.

See [Inference](inference.md) for commands and input schemas and
[Decoders](decoders.md) for training. The current encoder CLI supports wav2vec2;
other archived backbone implementations do not imply compatible inference.
