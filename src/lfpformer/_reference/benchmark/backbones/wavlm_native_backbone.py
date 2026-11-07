"""WavLM native backbone — actual WavLM architecture with contrastive SSL.

Uses the real WavLMModel (with gated relative position bias) instead of stuffing
WavLM weights into Wav2Vec2ForPreTraining. This preserves WavLM's key architectural
difference: gated relative position bias in attention.

Since HuggingFace has no WavLMForPreTraining, we build the pretraining wrapper
ourselves using:
  - WavLMModel: encoder with gated relative position bias
  - StableGumbelVectorQuantizer: codebook for contrastive targets (NaN-safe)
  - Contrastive loss + diversity loss (same formulation as wav2vec2)

Note: WavLM's native objective also includes masked speech denoising (mixing
input with noise/overlapping speech). This is not implemented here because the
concept doesn't directly transfer to LFP signals. Can be added as a future
extension with LFP-appropriate noise augmentation.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import WavLMConfig, WavLMModel
from transformers.models.wavlm.modeling_wavlm import WavLMGumbelVectorQuantizer
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices,
    _sample_negative_indices,
)

from backbones.base import BackboneBase


class StableGumbelVectorQuantizer(WavLMGumbelVectorQuantizer):
    """WavLM Gumbel vector quantizer with numerically stable Gumbel sampling.

    PyTorch's gumbel_softmax uses exponential_().log().neg() for Gumbel noise.
    With large tensors (B=64, seq=149, groups=2, vars=320 → 6M+ elements), the
    exponential distribution can produce values ≈0 on GPU, causing log(0)=-inf
    → -(-inf)=+inf → softmax(inf-inf)=NaN. This propagates through the
    straight-through estimator even with hard=True.

    Fix: Use clamped uniform sampling instead of exponential for Gumbel noise.
    """

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # Project to codevector dim
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.view(
            batch_size * sequence_length * self.num_groups, -1
        )

        if self.training:
            # Stable Gumbel softmax: clamp uniform samples to avoid log(0)
            codevector_probs = _stable_gumbel_softmax(
                hidden_states.float(), tau=self.temperature, hard=True
            )
            codevector_probs = codevector_probs.type_as(hidden_states)

            # Compute perplexity from soft distribution (no Gumbel noise)
            codevector_soft_dist = torch.softmax(
                hidden_states.view(
                    batch_size * sequence_length, self.num_groups, -1
                ).float(),
                dim=-1,
            )
            perplexity = self._compute_perplexity(codevector_soft_dist)
        else:
            # Eval: argmax (deterministic, no Gumbel noise)
            codevector_idx = hidden_states.argmax(dim=-1)
            codevector_probs = hidden_states.new_zeros(*hidden_states.shape).scatter_(
                -1, codevector_idx.view(-1, 1), 1.0
            )
            codevector_probs = codevector_probs.view(
                batch_size * sequence_length, self.num_groups, -1
            )
            perplexity = self._compute_perplexity(codevector_probs)

        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
        # Use probs to retrieve codevectors
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors
        codevectors = codevectors_per_group.view(
            batch_size * sequence_length, self.num_groups, self.num_vars, -1
        )
        codevectors = codevectors.sum(-2).view(batch_size, sequence_length, -1)

        return codevectors, perplexity


def _stable_gumbel_softmax(
    logits: torch.Tensor, tau: float = 1.0, hard: bool = False, eps: float = 1e-10
) -> torch.Tensor:
    """Numerically stable Gumbel softmax that avoids NaN from log(0).

    Standard PyTorch gumbel_softmax uses exponential_().log().neg() which can
    produce +inf when exponential samples hit 0 (subnormal float32). With large
    tensors (>1M elements), this happens reliably on GPU.
    """
    # Gumbel(0,1) = -log(-log(U)) where U ~ Uniform(0,1)
    # Clamp U away from 0 and 1 to avoid log(0) = -inf
    u = torch.empty_like(logits).uniform_().clamp_(eps, 1.0 - eps)
    gumbels = -torch.log(-torch.log(u))

    # Perturbed logits
    y = (logits + gumbels) / tau
    y_soft = y.softmax(dim=-1)

    if hard:
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        return y_hard - y_soft.detach() + y_soft

    return y_soft


class WavLMNativeBackbone(BackboneBase):
    """WavLM native backbone for SSL pretraining on LFP signals.

    Uses the actual WavLM encoder (with gated relative position bias) and
    implements contrastive pretraining (quantizer + negatives + diversity loss).

    Key difference from the old WavLMBackbone: that loaded WavLM weights into
    Wav2Vec2ForPreTraining, losing the gated relative position bias. This class
    uses WavLMModel directly, preserving the native architecture.
    """

    def __init__(self, config: dict):
        super().__init__()

        self._hidden_dim = config.get("hidden_size", 768)

        # Build WavLM config
        wavlm_config = WavLMConfig(
            hidden_size=self._hidden_dim,
            num_hidden_layers=config.get("num_hidden_layers", 12),
            num_attention_heads=config.get("num_attention_heads", 12),
            intermediate_size=config.get("intermediate_size", 3072),
            hidden_act="gelu",
            hidden_dropout=float(config.get("hidden_dropout", 0.1)),
            attention_dropout=float(config.get("attention_dropout", 0.1)),
            final_dropout=float(config.get("final_dropout", 0.0)),
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
            num_conv_pos_embedding_groups=16,
            apply_spec_augment=False,  # Match other backbones — SpecAugment not needed
            mask_time_prob=float(config.get("mask_time_prob", 0.065)),
            mask_time_length=int(config.get("mask_time_length", 10)),
            mask_time_min_masks=2,
            mask_feature_prob=0.0,
            mask_feature_length=10,
            # Quantizer / contrastive params
            num_codevectors_per_group=320,
            num_codevector_groups=2,
            contrastive_logits_temperature=0.1,
            num_negatives=int(config.get("num_negatives", 100)),
            codevector_dim=256,
            proj_codevector_dim=256,
            diversity_loss_weight=0.1,
            # WavLM-specific: gated relative position bias
            num_buckets=320,
            max_bucket_distance=800,
        )

        rand_init = config.get("rand_init", False)

        # Encoder: actual WavLM with gated relative position bias
        if rand_init:
            self.encoder = WavLMModel(wavlm_config)
        else:
            pretrained_id = config.get("pretrained_model", "microsoft/wavlm-base")
            self.encoder = WavLMModel.from_pretrained(
                pretrained_id, config=wavlm_config
            )
            print(f"[wavlm_native] Loaded encoder from {pretrained_id}")

        # Quantizer + projection layers (same as Wav2Vec2ForPreTraining)
        self.dropout_features = nn.Dropout(0.0)
        self.quantizer = StableGumbelVectorQuantizer(wavlm_config)

        # WavLMGumbelVectorQuantizer uses torch.FloatTensor (uninitialized memory)
        # for codevectors. In HuggingFace's pipeline, _init_weights handles this,
        # but since we create the quantizer standalone, we must initialize manually.
        nn.init.uniform_(self.quantizer.codevectors)
        nn.init.normal_(self.quantizer.weight_proj.weight, mean=0.0, std=1.0)
        nn.init.zeros_(self.quantizer.weight_proj.bias)

        self.project_hid = nn.Linear(wavlm_config.hidden_size, wavlm_config.proj_codevector_dim)
        self.project_q = nn.Linear(wavlm_config.codevector_dim, wavlm_config.proj_codevector_dim)

        # Store config for loss computation
        self._config = wavlm_config

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
                    self.encoder._get_feat_extract_output_lengths(raw_seq_len).item()
                )

        seq_len = self._cached_seq_len

        mask_time_indices = _compute_mask_indices(
            (batch_size, seq_len),
            mask_prob=self._config.mask_time_prob,
            mask_length=self._config.mask_time_length,
        )
        sampled_negatives = _sample_negative_indices(
            (batch_size, seq_len),
            num_negatives=self._config.num_negatives,
            mask_time_indices=mask_time_indices,
        )

        mask_time_indices = torch.from_numpy(mask_time_indices).to(device)
        sampled_negatives = torch.from_numpy(sampled_negatives).to(device)

        return mask_time_indices, sampled_negatives

    @staticmethod
    def _compute_contrastive_logits(target_features, negative_features,
                                     predicted_features, temperature):
        """Compute contrastive logits via cosine similarity."""
        target_features = torch.cat([target_features, negative_features], dim=0)
        logits = torch.cosine_similarity(
            predicted_features.float(), target_features.float(), dim=-1
        ).type_as(target_features)
        logits = logits / temperature
        return logits

    def ssl_forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """Contrastive SSL forward pass using native WavLM encoder.

        Follows the same loss formulation as Wav2Vec2ForPreTraining but
        uses WavLMModel (with gated relative position bias) as the encoder.
        """
        mask_time_indices, sampled_negatives = self._compute_masks(input_values)
        mask_time_indices_bool = mask_time_indices.to(torch.bool)

        # Encoder forward (with masking)
        outputs = self.encoder(
            input_values,
            mask_time_indices=mask_time_indices_bool,
        )

        # Project transformer output to codevector dim
        transformer_features = self.project_hid(outputs.last_hidden_state)

        # Quantize extracted features (pre-transformer)
        # WavLM quantizer doesn't take mask_time_indices (quantizes all positions)
        extract_features = self.dropout_features(outputs.extract_features)
        quantized_features, codevector_perplexity = self.quantizer(extract_features)
        quantized_features = quantized_features.to(self.project_q.weight.dtype)
        quantized_features = self.project_q(quantized_features)

        # Sample negatives
        batch_size, sequence_length, hidden_size = quantized_features.shape
        negative_quantized_features = quantized_features.view(-1, hidden_size)[
            sampled_negatives.long().view(-1)
        ]
        negative_quantized_features = negative_quantized_features.view(
            batch_size, sequence_length, -1, hidden_size
        ).permute(2, 0, 1, 3)

        # Contrastive logits
        logits = self._compute_contrastive_logits(
            quantized_features[None, :],
            negative_quantized_features,
            transformer_features,
            self._config.contrastive_logits_temperature,
        )

        # Mask out negatives identical to positive (low codebook utilization)
        neg_is_pos = (quantized_features == negative_quantized_features).all(-1)
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")

        # Contrastive loss
        logits = logits.transpose(0, 2).reshape(-1, logits.size(0))
        target = ((1 - mask_time_indices.long()) * -100).transpose(0, 1).flatten()
        contrastive_loss = nn.functional.cross_entropy(logits.float(), target, reduction="sum")

        # Diversity loss
        num_codevectors = self._config.num_codevectors_per_group * self._config.num_codevector_groups
        diversity_loss = ((num_codevectors - codevector_perplexity) / num_codevectors) * mask_time_indices.sum()

        # Total loss
        loss = contrastive_loss + self._config.diversity_loss_weight * diversity_loss

        return loss

    def encode(self, input_values: torch.Tensor) -> torch.Tensor:
        """Extract embeddings using the WavLM encoder."""
        outputs = self.encoder(input_values)
        hidden_states = outputs.last_hidden_state  # [B, seq_len, hidden_dim]
        return hidden_states.mean(dim=1)  # [B, hidden_dim]
