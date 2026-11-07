"""WavLM backbone — uses WavLM encoder with wav2vec2's contrastive pretraining.

WavLM has no ForPreTraining class in HuggingFace, so we load the WavLM encoder
into a Wav2Vec2ForPreTraining wrapper to get contrastive SSL. This gives us
WavLM's gated relative position bias with wav2vec2's pretraining infrastructure.
"""

from typing import Optional

import torch

from transformers import Wav2Vec2Config, Wav2Vec2ForPreTraining, WavLMModel
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices,
    _sample_negative_indices,
)

from backbones.base import BackboneBase


class WavLMBackbone(BackboneBase):
    """WavLM backbone for SSL pretraining on LFP signals.

    Strategy: Load WavLM pretrained encoder weights into Wav2Vec2ForPreTraining.
    The architectures are nearly identical (WavLM extends wav2vec2), so the
    encoder weights transfer cleanly. This gives us:
    - WavLM's pretrained features (94k hours, gated relative position bias)
    - wav2vec2's contrastive SSL training infrastructure
    """

    def __init__(self, config: dict):
        super().__init__()

        self._hidden_dim = config.get("hidden_size", 768)

        # Build wav2vec2 config (compatible with WavLM encoder)
        w2v2_config = Wav2Vec2Config(
            vocab_size=32,
            hidden_size=self._hidden_dim,
            num_hidden_layers=config.get("num_hidden_layers", 12),
            num_attention_heads=config.get("num_attention_heads", 12),
            intermediate_size=config.get("intermediate_size", 3072),
            hidden_act="gelu",
            hidden_dropout=config.get("hidden_dropout", 0.1),
            attention_dropout=config.get("attention_dropout", 0.1),
            final_dropout=config.get("final_dropout", 0.0),
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            feat_extract_norm="group",
            feat_proj_dropout=0.0,
            feat_extract_activation="gelu",
            conv_dim=(512, 512, 512, 512, 512, 512, 512),
            conv_stride=(5, 2, 2, 2, 2, 2, 2),
            conv_kernel=(10, 3, 3, 3, 3, 2, 2),
            conv_bias=False,
            num_conv_pos_embeddings=128,
            num_conv_pos_embeddings_groups=16,
            apply_spec_augment=True,
            mask_time_prob=config.get("mask_time_prob", 0.065),
            mask_time_length=config.get("mask_time_length", 10),
            mask_time_min_masks=2,
            mask_feature_prob=0.0,
            mask_feature_length=10,
            num_codevectors_per_group=320,
            num_codevector_groups=2,
            contrastive_logits_temperature=0.1,
            num_negatives=config.get("num_negatives", 100),
            codevector_dim=256,
            proj_codevector_dim=256,
            diversity_loss_weight=0.1,
        )

        rand_init = config.get("rand_init", False)

        # Create wav2vec2 pretraining model
        self.model = Wav2Vec2ForPreTraining(w2v2_config)

        # Load WavLM pretrained encoder weights if not random init
        if not rand_init:
            pretrained_id = config.get("pretrained_model", "microsoft/wavlm-base")
            wavlm_encoder = WavLMModel.from_pretrained(pretrained_id)

            # Transfer encoder weights (shared architecture)
            w2v2_encoder = self.model.wav2vec2
            wavlm_state = wavlm_encoder.state_dict()
            w2v2_state = w2v2_encoder.state_dict()

            # Load matching keys
            loaded = 0
            for key in w2v2_state:
                if key in wavlm_state and w2v2_state[key].shape == wavlm_state[key].shape:
                    w2v2_state[key] = wavlm_state[key]
                    loaded += 1

            w2v2_encoder.load_state_dict(w2v2_state)
            print(f"[WavLM] Loaded {loaded}/{len(w2v2_state)} encoder weights from {pretrained_id}")

            del wavlm_encoder  # free memory

        self._cached_seq_len: Optional[int] = None

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def needs_resampling(self) -> bool:
        return True

    def _compute_masks(self, input_values: torch.Tensor) -> tuple:
        batch_size, raw_seq_len = input_values.shape
        device = input_values.device

        if self._cached_seq_len is None:
            with torch.no_grad():
                self._cached_seq_len = int(
                    self.model._get_feat_extract_output_lengths(raw_seq_len).item()
                )

        seq_len = self._cached_seq_len

        mask_time_indices = _compute_mask_indices(
            (batch_size, seq_len),
            mask_prob=self.model.config.mask_time_prob,
            mask_length=self.model.config.mask_time_length,
        )
        sampled_negatives = _sample_negative_indices(
            (batch_size, seq_len),
            num_negatives=self.model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )

        mask_time_indices = torch.from_numpy(mask_time_indices).to(device)
        sampled_negatives = torch.from_numpy(sampled_negatives).to(device)

        return mask_time_indices, sampled_negatives

    def ssl_forward(self, input_values: torch.Tensor) -> torch.Tensor:
        mask_time_indices, sampled_negatives = self._compute_masks(input_values)
        outputs = self.model(
            input_values=input_values,
            mask_time_indices=mask_time_indices,
            sampled_negative_indices=sampled_negatives,
        )
        return outputs.loss

    def encode(self, input_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state  # [B, seq_len, hidden_dim]
        return hidden_states.mean(dim=1)  # [B, hidden_dim]
