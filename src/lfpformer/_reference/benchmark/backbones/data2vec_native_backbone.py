"""data2vec 2.0 native backbone — EMA teacher-student with L2 regression loss.

Implements the native data2vec SSL objective (Baevski et al., 2022):
  1. Student: Data2VecAudioModel with masking, predicts teacher targets via ConvDecoder
  2. Teacher: EMA copy of student (no grad), produces targets by averaging top-K layers
  3. Loss: MSE (L2) between student predictions and instance-normalized teacher targets
  4. No quantizer, no negative samples, no codebook — pure regression

This contrasts with Experiment Group 1 where data2vec encoder weights are loaded
into Wav2Vec2ForPreTraining and trained with the contrastive objective.
"""

import copy
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Data2VecAudioConfig, Data2VecAudioModel
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

from backbones.base import BackboneBase


class ConvDecoder(nn.Module):
    """Lightweight 1D convolutional decoder (data2vec 2.0 style).

    Maps student encoder hidden states to the target space for regression.
    Uses depthwise-separable convolutions for efficiency.
    """

    def __init__(self, hidden_dim: int = 768, layers: int = 4,
                 kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()

        convs = []
        for _ in range(layers):
            convs.extend([
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size,
                          padding=kernel_size // 2, groups=hidden_dim),  # depthwise
                nn.Conv1d(hidden_dim, hidden_dim, 1),  # pointwise
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            ])
        self.convs = nn.ModuleList(convs)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, T, C] hidden states from student encoder

        Returns:
            [B, T, C] predicted targets
        """
        # Conv layers expect [B, C, T]
        h = x.transpose(1, 2)
        for layer in self.convs:
            if isinstance(layer, nn.LayerNorm):
                # LayerNorm expects [B, T, C]
                h = layer(h.transpose(1, 2)).transpose(1, 2)
            elif isinstance(layer, (nn.GELU, nn.Dropout)):
                h = layer(h)
            else:
                h = layer(h)
        # Back to [B, T, C]
        h = h.transpose(1, 2)
        return self.proj(h)


class Data2VecNativeBackbone(BackboneBase):
    """data2vec 2.0 native SSL backbone.

    EMA teacher-student architecture with L2 regression loss.
    The teacher is an exponential moving average of the student,
    producing targets by averaging the top-K transformer layers.
    """

    def __init__(self, config: dict):
        super().__init__()

        self._hidden_dim = config.get("hidden_size", 768)

        # data2vec 2.0 specific params
        self.ema_decay_init = float(config.get("ema_decay", 0.999))
        self.ema_end_decay = float(config.get("ema_end_decay", 0.9999))
        self.ema_anneal_end_step = int(config.get("ema_anneal_end_step", 30000))
        self.average_top_k_layers = int(config.get("average_top_k_layers", 8))

        # Masking params (stored for ssl_forward)
        self.mask_time_prob = float(config.get("mask_time_prob", 0.065))
        self.mask_time_length = int(config.get("mask_time_length", 10))

        # Build Data2VecAudioConfig
        audio_config = Data2VecAudioConfig(
            hidden_size=self._hidden_dim,
            num_hidden_layers=config.get("num_hidden_layers", 12),
            num_attention_heads=config.get("num_attention_heads", 12),
            intermediate_size=config.get("intermediate_size", 3072),
            hidden_act="gelu",
            hidden_dropout=float(config.get("hidden_dropout", 0.1)),
            attention_dropout=float(config.get("attention_dropout", 0.1)),
            activation_dropout=float(config.get("attention_dropout", 0.1)),
            final_dropout=float(config.get("final_dropout", 0.0)),
            layerdrop=0.0,  # Must be 0 — teacher and student must use same layers
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            feat_proj_dropout=0.0,
            feat_extract_activation="gelu",
            conv_dim=(512, 512, 512, 512, 512, 512, 512),
            conv_stride=(5, 2, 2, 2, 2, 2, 2),
            conv_kernel=(10, 3, 3, 3, 3, 2, 2),
            conv_bias=False,
            num_conv_pos_embeddings=5,
            num_conv_pos_embedding_groups=16,
            apply_spec_augment=False,
            mask_time_prob=self.mask_time_prob,
            mask_time_length=self.mask_time_length,
            mask_time_min_masks=2,
            mask_feature_prob=0.0,
        )

        rand_init = config.get("rand_init", False)

        # Student model
        if rand_init:
            self.student = Data2VecAudioModel(audio_config)
        else:
            pretrained_id = config.get("pretrained_model", "facebook/data2vec-audio-base")
            self.student = Data2VecAudioModel.from_pretrained(
                pretrained_id, config=audio_config
            )
            print(f"[data2vec_native] Loaded student from {pretrained_id}")

        # Teacher model — deep copy of student, frozen
        self.teacher = copy.deepcopy(self.student)
        self.teacher.requires_grad_(False)

        # Conv decoder (data2vec 2.0)
        self.decoder = ConvDecoder(
            hidden_dim=self._hidden_dim,
            layers=4,
            kernel_size=5,
            dropout=float(config.get("hidden_dropout", 0.1)),
        )

        # Step counter for EMA annealing
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))

        # Cache sequence length for mask computation
        self._cached_seq_len: Optional[int] = None

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def needs_resampling(self) -> bool:
        return True  # LFP is 1250 Hz, model expects 16000 Hz

    def _get_annealed_decay(self) -> float:
        """Linear anneal of EMA decay from ema_decay_init to ema_end_decay."""
        step = self.num_updates.item()
        if step >= self.ema_anneal_end_step:
            return self.ema_end_decay
        r = step / self.ema_anneal_end_step
        return self.ema_decay_init + r * (self.ema_end_decay - self.ema_decay_init)

    @torch.no_grad()
    def _update_ema(self):
        """Update teacher weights via exponential moving average of student."""
        decay = self._get_annealed_decay()
        for s_param, t_param in zip(self.student.parameters(), self.teacher.parameters()):
            t_param.data.mul_(decay).add_(s_param.data, alpha=1 - decay)
        self.num_updates += 1

    def ssl_forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """Native data2vec 2.0 SSL forward pass.

        Args:
            input_values: [B, T] raw waveform (already resampled to 16kHz)

        Returns:
            Scalar MSE loss at masked positions
        """
        B, T = input_values.shape
        device = input_values.device

        # Compute output sequence length
        if self._cached_seq_len is None:
            with torch.no_grad():
                self._cached_seq_len = int(
                    self.student._get_feat_extract_output_lengths(T).item()
                )
        seq_len = self._cached_seq_len

        # Compute mask indices
        mask_time_indices = _compute_mask_indices(
            (B, seq_len),
            mask_prob=self.mask_time_prob,
            mask_length=self.mask_time_length,
        )
        mask_time_indices = torch.from_numpy(mask_time_indices).to(device)

        # 1. Teacher forward (no grad, no masking, output all hidden states)
        with torch.no_grad():
            teacher_out = self.teacher(
                input_values,
                output_hidden_states=True,
            )
            # Average top-K transformer layer outputs for target
            # hidden_states: tuple of (embedding + 12 layers) = 13 tensors
            top_k_states = teacher_out.hidden_states[-self.average_top_k_layers:]
            target = torch.stack(list(top_k_states)).mean(dim=0)  # [B, seq_len, 768]
            # Instance normalize target (data2vec 2.0)
            target = F.instance_norm(target.transpose(1, 2)).transpose(1, 2)

        # 2. Student forward (with masking)
        student_out = self.student(
            input_values,
            mask_time_indices=mask_time_indices,
        )
        # Decode student hidden states to target space
        predicted = self.decoder(student_out.last_hidden_state)  # [B, seq_len, 768]

        # 3. L2 loss at masked positions only
        mask = mask_time_indices.bool()  # [B, seq_len]
        loss = F.mse_loss(predicted[mask], target[mask])

        # 4. Update teacher EMA (only during training)
        if self.training:
            self._update_ema()

        return loss

    def encode(self, input_values: torch.Tensor) -> torch.Tensor:
        """Extract embeddings using the student encoder.

        Args:
            input_values: [B, T] raw waveform

        Returns:
            [B, hidden_dim] mean-pooled representation
        """
        outputs = self.student(input_values)
        hidden_states = outputs.last_hidden_state  # [B, seq_len, hidden_dim]
        return hidden_states.mean(dim=1)  # [B, hidden_dim]

    def get_optimizer_params(self, lr: float) -> List[Dict[str, Any]]:
        """Return optimizer params — student + decoder only (teacher excluded)."""
        params = []
        for p in self.student.parameters():
            if p.requires_grad:
                params.append(p)
        params.extend(self.decoder.parameters())
        return [{"params": params, "lr": lr}]

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count parameters — student + decoder only (teacher excluded)."""
        if trainable_only:
            student_params = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
            decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
            return student_params + decoder_params
        student_params = sum(p.numel() for p in self.student.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        return student_params + decoder_params
