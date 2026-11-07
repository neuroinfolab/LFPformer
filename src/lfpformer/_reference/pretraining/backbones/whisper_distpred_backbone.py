"""Whisper pairwise distance prediction backbone — predict inter-electrode distances.

Instead of masked reconstruction (Data2Vec-style) or contrastive learning, this
backbone uses pairwise distance prediction as the sole SSL objective: given two
LFP embeddings from different electrodes, predict the physical Euclidean distance
between them via a regression MLP.

Architecture:
    LFP chunk @ 16kHz → WhisperFeatureExtractor → mel [B, 80, 3000]
                                                        |
                                                  Student encoder
                                                  (trainable)
                                                  last_hidden_state
                                                        |
                                                  mean-pool → z [B, D]
                                                        |
                              For B random pairs (i,j):
                                [z_i; z_j] → MLP → predicted log-distance
                              Target: log(1 + ||coord_i - coord_j||₂)
                              Loss: MSE(predicted, target)
"""

import copy
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import WhisperConfig, WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from backbones.base import BackboneBase


class WhisperDistPredBackbone(BackboneBase):
    """Whisper encoder with pairwise distance prediction SSL.

    The student encoder produces mean-pooled embeddings. For each batch,
    B random pairs are sampled and their concatenated embeddings are passed
    through an MLP to predict the log-scaled Euclidean distance between
    their CCF coordinates.

    EMA teacher is kept for future extension but not used in the loss.
    """

    ENCODER_FRAMES = 1500  # Whisper encoder output length for 30s

    def __init__(self, config: dict):
        super().__init__()

        self._hidden_dim = config.get("hidden_size", 512)
        model_id = config.get("pretrained_model", "openai/whisper-base")
        rand_init = config.get("rand_init", False)

        # EMA params (kept for teacher stability)
        self.ema_decay_init = float(config.get("ema_decay", 0.999))
        self.ema_end_decay = float(config.get("ema_end_decay", 0.9999))
        self.ema_anneal_end_step = int(config.get("ema_anneal_end_step", 500))
        self.average_top_k_layers = int(config.get("average_top_k_layers", 6))

        # Build Whisper encoder (student)
        if rand_init or model_id is None:
            self.feature_extractor = WhisperFeatureExtractor(
                feature_size=80,
                sampling_rate=16000,
                hop_length=160,
                chunk_length=30,
                n_fft=400,
            )
            whisper_config = WhisperConfig(
                d_model=self._hidden_dim,
                encoder_layers=config.get("num_hidden_layers", 6),
                encoder_attention_heads=config.get("num_attention_heads", 8),
                encoder_ffn_dim=config.get("intermediate_size", 2048),
                dropout=float(config.get("hidden_dropout", 0.1)),
                attention_dropout=float(config.get("attention_dropout", 0.0)),
                activation_dropout=float(config.get("final_dropout", 0.0)),
            )
            self.student = WhisperEncoder(whisper_config)
        else:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
            whisper_config = WhisperConfig.from_pretrained(model_id)
            whisper_config.dropout = float(config.get("hidden_dropout", whisper_config.dropout))
            whisper_config.attention_dropout = float(config.get("attention_dropout", whisper_config.attention_dropout))
            whisper_config.activation_dropout = float(config.get("final_dropout", whisper_config.activation_dropout))
            self.student = WhisperEncoder(whisper_config)
            from transformers import WhisperModel
            pretrained = WhisperModel.from_pretrained(model_id)
            self.student.load_state_dict(pretrained.encoder.state_dict())
            self._hidden_dim = whisper_config.d_model
            del pretrained
            print(f"[whisper_distpred] Loaded student from {model_id}")

        # Teacher — deep copy of student, frozen + eval mode (no dropout in targets)
        self.teacher = copy.deepcopy(self.student)
        self.teacher.requires_grad_(False)
        self.teacher.eval()

        # Pairwise distance prediction MLP
        # Input: [z_i; z_j] concatenated embeddings (2 * D)
        # Output: scalar predicted log-distance
        D = self._hidden_dim
        self.dist_pred_head = nn.Sequential(
            nn.Linear(2 * D, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Step counter for EMA annealing
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))

    def train(self, mode: bool = True):
        """Override to keep teacher permanently in eval mode."""
        super().train(mode)
        self.teacher.eval()
        return self

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def encoder(self):
        """Alias for student encoder (compatibility with finetune code)."""
        return self.student

    @property
    def needs_resampling(self) -> bool:
        return True  # Expects 16kHz input

    @property
    def needs_normalization(self) -> bool:
        return True  # LFP needs ×3000 global scale for Whisper's log-mel

    def _compute_mel(self, input_values: torch.Tensor) -> torch.Tensor:
        """Convert raw waveform to log-mel spectrogram.

        Args:
            input_values: [B, T] raw waveforms (T ~ 48000 for 3s @ 16kHz)

        Returns:
            [B, 80, 3000] mel spectrograms
        """
        device = input_values.device
        batch_np = input_values.cpu().numpy()

        mel_features = self.feature_extractor(
            batch_np,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            max_length=16000 * 30,
            truncation=True,
        )
        return mel_features.input_features.to(device)  # [B, 80, 3000]

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

    def _compute_pairwise_dist_loss(
        self, chunk_embeds: torch.Tensor, coords: torch.Tensor,
    ) -> torch.Tensor:
        """Pairwise distance prediction loss.

        Samples B random pairs from the batch, concatenates their embeddings,
        and regresses the log-scaled Euclidean distance between CCF coordinates.

        Args:
            chunk_embeds: [B, D] mean-pooled student encoder output
            coords: [B, 3] raw CCF coordinates in um

        Returns:
            Scalar MSE loss
        """
        B = chunk_embeds.shape[0]
        if B < 2:
            return (chunk_embeds * 0.0).sum()  # differentiable zero

        # Sample B random pairs (i, j) where i != j
        idx_i = torch.randint(0, B, (B,), device=chunk_embeds.device)
        idx_j = torch.randint(0, B - 1, (B,), device=chunk_embeds.device)
        # Shift j >= i to avoid self-pairs
        idx_j = idx_j + (idx_j >= idx_i).long()

        # Concatenate pair embeddings
        z_i = chunk_embeds[idx_i]  # [B, D]
        z_j = chunk_embeds[idx_j]  # [B, D]
        pair_embeds = torch.cat([z_i, z_j], dim=-1)  # [B, 2*D]

        # Predict distance
        pred_dist = self.dist_pred_head(pair_embeds).squeeze(-1)  # [B]

        # Target: log(1 + euclidean_distance)
        with torch.no_grad():
            coord_diff = coords[idx_i] - coords[idx_j]  # [B, 3]
            euclidean_dist = torch.sqrt((coord_diff ** 2).sum(dim=-1) + 1e-8)  # [B]
            target_dist = torch.log1p(euclidean_dist)  # [B]

        loss = F.mse_loss(pred_dist, target_dist)
        return loss

    def ssl_forward(self, input_values: torch.Tensor,
                    coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pairwise distance prediction SSL forward pass.

        Args:
            input_values: [B, T] raw waveform (already resampled to 16kHz)
            coords: [B, 3] CCF coordinates (required for this backbone)

        Returns:
            Scalar loss
        """
        mel = self._compute_mel(input_values)  # [B, 80, 3000]

        # Student forward
        student_out = self.student(mel)
        student_hidden = student_out.last_hidden_state  # [B, 1500, D]

        # Mean-pool for per-chunk embeddings
        chunk_embeds = student_hidden.mean(dim=1)  # [B, D]

        # Pairwise distance prediction loss
        if coords is not None:
            loss = self._compute_pairwise_dist_loss(chunk_embeds, coords)
        else:
            loss = (chunk_embeds * 0.0).sum()  # differentiable zero

        # Update teacher EMA
        if self.training:
            self._update_ema()

        return loss

    def ssl_forward_detailed(self, input_values: torch.Tensor,
                              coords: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Pairwise distance prediction with detailed outputs for logging.

        Args:
            input_values: [B, T] raw waveform (already resampled to 16kHz)
            coords: [B, 3] CCF coordinates

        Returns:
            dict compatible with train_ssl.py logging expectations
        """
        mel = self._compute_mel(input_values)  # [B, 80, 3000]

        student_out = self.student(mel)
        student_hidden = student_out.last_hidden_state  # [B, 1500, D]
        chunk_embeds = student_hidden.mean(dim=1)  # [B, D]

        if coords is not None:
            loss = self._compute_pairwise_dist_loss(chunk_embeds, coords)
        else:
            loss = (chunk_embeds * 0.0).sum()  # differentiable zero

        if self.training:
            self._update_ema()

        zero = torch.tensor(0.0, device=input_values.device)
        B = student_hidden.shape[0]
        seq_len = student_hidden.shape[1]
        D = student_hidden.shape[2]
        # Dummy mask/predicted/target for compatibility with reconstruction logging
        dummy_mask = torch.zeros(B, seq_len, device=input_values.device, dtype=torch.bool)
        return {
            "loss": loss,
            "ssl_loss": loss.detach(),
            "xyz_loss": zero,
            "dist_loss": zero,
            "mask_bool": dummy_mask,
            "predicted": student_hidden.detach(),
            "target": student_hidden.detach(),
        }

    def encode(self, input_values: torch.Tensor) -> torch.Tensor:
        """Extract mean-pooled embeddings.

        Args:
            input_values: [B, T] raw waveform (3s chunks @ 16kHz)

        Returns:
            [B, hidden_dim] per-chunk embeddings
        """
        mel = self._compute_mel(input_values)  # [B, 80, 3000]
        encoder_out = self.student(mel).last_hidden_state  # [B, 1500, D]
        return encoder_out.mean(dim=1)  # [B, D]

    def get_optimizer_params(self, lr: float) -> List[Dict[str, Any]]:
        """Return optimizer params — student + dist_pred_head (teacher excluded)."""
        encoder_lr = lr * 0.1  # Lower LR for pretrained encoder
        return [
            {"params": [p for p in self.student.parameters() if p.requires_grad], "lr": encoder_lr},
            {"params": self.dist_pred_head.parameters(), "lr": lr},
        ]

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count parameters — student + dist_pred_head (teacher excluded)."""
        if trainable_only:
            student_p = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
            head_p = sum(p.numel() for p in self.dist_pred_head.parameters() if p.requires_grad)
            return student_p + head_p
        student_p = sum(p.numel() for p in self.student.parameters())
        head_p = sum(p.numel() for p in self.dist_pred_head.parameters())
        return student_p + head_p
