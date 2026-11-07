"""Whisper self-distillation backbone — EMA teacher-student with L2 regression.

Combines Whisper's mel-spectrogram frontend with Data2Vec's EMA teacher-student
SSL objective. This avoids the contrastive collapse seen with Wav2Vec2 by using
a pure regression target from instance-normalized averaged top-K teacher layers.

Optional spatial auxiliary losses (config-driven, can be combined):
  - XYZ regression: predicts electrode CCF coordinates from embeddings
  - dist_contrastive: soft contrastive loss pulling nearby electrodes together
  - dist_prediction: MLP regresses pairwise log-distance between electrode pairs

Architecture:
    LFP chunk @ 16kHz → WhisperFeatureExtractor → mel [B, 80, 3000]
                                                        |
                            +--------------------------------------+
                            v                                      v
                      Teacher encoder                       Student encoder
                      (frozen, EMA)                         (trainable, masked)
                      output_hidden_states=True              last_hidden_state
                            |                                      |
                            v                                      v
                      avg top-K layers                     ConvDecoder → predictions
                      instance_norm                               |
                            |                                      |
                            +------- MSE loss at masked pos ------+
                                              +
                            λ_xyz * MSE(xyz_head(embed), coords)
                                              +
                            λ_dist * dist_contrastive(embed, coords)
                                              +
                            λ_distpred * MSE(dist_pred_head([z_i;z_j]), log(1+d_ij))
"""

import copy
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import WhisperConfig, WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder, BaseModelOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

from backbones.base import BackboneBase


class FlexibleWhisperEncoder(WhisperEncoder):
    """WhisperEncoder that accepts variable-length mel input (not just 3000 frames).

    HuggingFace WhisperEncoder.forward() has a hardcoded check requiring mel input
    of exactly 3000 frames (30s). This subclass removes that check and slices
    positional embeddings to match the actual sequence length.

    This allows processing 3s LFP chunks directly (300 mel frames → 150 encoder frames)
    without padding to 30s, avoiding the padding dilution problem.
    """

    def forward(self, input_features, attention_mask=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, **kwargs):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Conv frontend — works with any length
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)  # [B, T, D]

        # Positional embeddings — slice to actual sequence length (not hardcoded 1500)
        seq_len = inputs_embeds.shape[1]
        positions = torch.arange(seq_len, device=inputs_embeds.device)
        hidden_states = inputs_embeds + self.embed_positions(positions)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    to_drop = True
            if to_drop:
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states, attention_mask=None, output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class ConvDecoder(nn.Module):
    """Lightweight 1D convolutional decoder (data2vec 2.0 style).

    Maps student encoder hidden states to the target space for regression.
    Uses depthwise-separable convolutions for efficiency.
    """

    def __init__(self, hidden_dim: int = 512, layers: int = 4,
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
        h = x.transpose(1, 2)  # [B, C, T]
        for layer in self.convs:
            if isinstance(layer, nn.LayerNorm):
                h = layer(h.transpose(1, 2)).transpose(1, 2)
            elif isinstance(layer, (nn.GELU, nn.Dropout)):
                h = layer(h)
            else:
                h = layer(h)
        h = h.transpose(1, 2)  # [B, T, C]
        return self.proj(h)


class WhisperDistillBackbone(BackboneBase):
    """Whisper encoder with Data2Vec-style EMA teacher-student SSL.

    The teacher is an exponential moving average of the student encoder,
    producing targets by averaging the top-K transformer layers with
    instance normalization. The student predicts these targets at masked
    positions via a ConvDecoder.

    Each 3s LFP chunk is padded to 30s by WhisperFeatureExtractor (Whisper
    requires exactly 3000 mel frames). The full 1500-frame encoder output
    is mean-pooled for downstream embeddings.

    Optional XYZ auxiliary loss forces spatially-grounded representations
    by predicting electrode coordinates from the mean-pooled embeddings.
    """

    ENCODER_FRAMES = 1500  # Whisper encoder output length for 30s (overridden if flexible_length)

    def __init__(self, config: dict):
        super().__init__()

        self._hidden_dim = config.get("hidden_size", 512)
        model_id = config.get("pretrained_model", "openai/whisper-base")
        rand_init = config.get("rand_init", False)

        # Flexible length: use 3s input without padding to 30s.
        # Reduces encoder frames from 1500 to ~150, eliminating padding dilution.
        self.flexible_length = bool(config.get("flexible_length", False))

        # EMA params
        self.ema_decay_init = float(config.get("ema_decay", 0.999))
        self.ema_end_decay = float(config.get("ema_end_decay", 0.9999))
        self.ema_anneal_end_step = int(config.get("ema_anneal_end_step", 500))
        self.average_top_k_layers = int(config.get("average_top_k_layers", 6))

        # Masking params
        self.mask_time_prob = float(config.get("mask_time_prob", 0.3))
        self.mask_time_length = int(config.get("mask_time_length", 10))

        # Build Whisper encoder (student)
        # Use FlexibleWhisperEncoder when flexible_length=True to accept variable-length mel
        EncoderClass = FlexibleWhisperEncoder if self.flexible_length else WhisperEncoder
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
            self.student = EncoderClass(whisper_config)
        else:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
            whisper_config = WhisperConfig.from_pretrained(model_id)
            # Apply dropout overrides from config
            whisper_config.dropout = float(config.get("hidden_dropout", whisper_config.dropout))
            whisper_config.attention_dropout = float(config.get("attention_dropout", whisper_config.attention_dropout))
            whisper_config.activation_dropout = float(config.get("final_dropout", whisper_config.activation_dropout))
            self.student = EncoderClass(whisper_config)
            # Load pretrained encoder weights
            from transformers import WhisperModel
            pretrained = WhisperModel.from_pretrained(model_id)
            self.student.load_state_dict(pretrained.encoder.state_dict())
            self._hidden_dim = whisper_config.d_model
            del pretrained
            print(f"[whisper_distill] Loaded student from {model_id} (flexible_length={self.flexible_length})")

        # Teacher — deep copy of student, frozen + eval mode (no dropout in targets)
        self.teacher = copy.deepcopy(self.student)
        self.teacher.requires_grad_(False)
        self.teacher.eval()

        # Partial encoder freezing: freeze all but top-k transformer layers.
        # unfreeze_top_k_layers=num_hidden_layers (default) → fully unfrozen (original behaviour)
        # unfreeze_top_k_layers=0 → whole encoder frozen; only decoder/heads train
        self._num_encoder_layers = int(config.get("num_hidden_layers", 6))
        unfreeze_k = int(config.get("unfreeze_top_k_layers", self._num_encoder_layers))
        unfreeze_k = max(0, min(unfreeze_k, self._num_encoder_layers))
        if unfreeze_k < self._num_encoder_layers:
            self.student.requires_grad_(False)
            if unfreeze_k > 0:
                for layer in self.student.layers[-unfreeze_k:]:
                    layer.requires_grad_(True)
                self.student.layer_norm.requires_grad_(True)
            print(f"[whisper_distill] Partial freeze: top {unfreeze_k}/{self._num_encoder_layers} layers unfrozen")

        # Mask embedding (learnable vector replacing masked positions)
        self.mask_emb = nn.Parameter(torch.zeros(self._hidden_dim))
        nn.init.normal_(self.mask_emb, std=0.02)

        # Conv decoder (data2vec 2.0 style)
        self.decoder = ConvDecoder(
            hidden_dim=self._hidden_dim,
            layers=4,
            kernel_size=5,
            dropout=float(config.get("hidden_dropout", 0.1)),
        )

        # XYZ coordinate prediction head (anti-collapse auxiliary loss)
        self.xyz_lambda = float(config.get("xyz_lambda", 0.0))
        self.xyz_head = nn.Sequential(
            nn.Linear(self._hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )
        # Register normalization stats (computed on first batch if needed)
        self.register_buffer("xyz_mean", torch.zeros(3))
        self.register_buffer("xyz_std", torch.ones(3))
        self.register_buffer("xyz_stats_initialized", torch.tensor(False))

        # Soft distance-weighted contrastive loss
        self.dist_lambda = float(config.get("dist_lambda", 0.0))
        self.dist_temperature = float(config.get("dist_temperature", 0.2))
        self.dist_sigma = float(config.get("dist_sigma", 500.0))  # µm

        # Memory bank (MoCo-style queue) for dist_contrastive
        # Extends the effective negative pool to B + queue_size samples per step.
        # queue_size=0 disables (default — falls back to batch-only).
        self.queue_size = int(config.get("queue_size", 0))
        if self.queue_size > 0:
            D = self._hidden_dim
            self.register_buffer("queue", F.normalize(torch.randn(self.queue_size, D), dim=1))
            self.register_buffer("queue_coords", torch.zeros(self.queue_size, 3))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("queue_filled", torch.zeros(1, dtype=torch.long))

        # Pairwise distance prediction loss (regression MLP)
        self.distpred_lambda = float(config.get("distpred_lambda", 0.0))

        # Normalized loss weights (HP search: ssl+distc+distpred sum to 1)
        # If any *_raw param is set, weights are normalized at forward time.
        # Falls back to legacy dist_lambda/distpred_lambda if raw params absent.
        self.ssl_lambda_raw = float(config.get("ssl_lambda_raw", 0.0))
        self.dist_lambda_raw = float(config.get("dist_lambda_raw", 0.0))
        self.distpred_lambda_raw = float(config.get("distpred_lambda_raw", 0.0))
        self.normalize_loss_weights = (
            self.ssl_lambda_raw > 0
            or self.dist_lambda_raw > 0
            or self.distpred_lambda_raw > 0
        )

        _distpred_active = self.distpred_lambda > 0 or self.distpred_lambda_raw > 0
        if _distpred_active:
            D = self._hidden_dim
            self.dist_pred_head = nn.Sequential(
                nn.Linear(2 * D, 256),
                nn.GELU(),
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

        # VICReg: variance + covariance regularization (Bardes et al. 2021).
        # Prevents collapse by enforcing diversity across embedding dimensions.
        self.vicreg_var_coeff = float(config.get("vicreg_var_coeff", 0.0))
        self.vicreg_cov_coeff = float(config.get("vicreg_cov_coeff", 0.0))
        self.vicreg_gamma = float(config.get("vicreg_gamma", 1.0))

        # Masked reconstruction loss (single-pass, no teacher — MAE/BERT style).
        # Targets = conv layer output (stop-grad, before transformer).
        # student(masked mel) → recon_decoder → MSE vs conv features at masked positions.
        self.recon_lambda = float(config.get("recon_lambda", 0.0))
        if self.recon_lambda > 0:
            self.recon_decoder = nn.Linear(self._hidden_dim, self._hidden_dim)

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

        When flexible_length=False (default): pads to 30s → mel [B, 80, 3000].
        When flexible_length=True: no padding → mel [B, 80, ~300] for 3s input.

        Args:
            input_values: [B, T] raw waveforms (T ≈ 48000 for 3s @ 16kHz)

        Returns:
            [B, 80, mel_frames] mel spectrograms
        """
        device = input_values.device
        batch_np = input_values.cpu().numpy()

        if self.flexible_length:
            # No padding — compute mel at actual signal length
            mel_features = self.feature_extractor(
                batch_np,
                sampling_rate=16000,
                return_tensors="pt",
                padding=False,
                truncation=False,
            )
        else:
            # Original: pad to 30s
            mel_features = self.feature_extractor(
                batch_np,
                sampling_rate=16000,
                return_tensors="pt",
                padding="max_length",
                max_length=16000 * 30,
                truncation=True,
            )
        return mel_features.input_features.to(device)

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

    @torch.no_grad()
    def _init_xyz_stats(self, coords: torch.Tensor):
        """Compute z-score normalization stats from first batch of coords."""
        self.xyz_mean.copy_(coords.mean(dim=0))
        std = coords.std(dim=0)
        std[std < 1e-6] = 1.0  # prevent division by zero
        self.xyz_std.copy_(std)
        self.xyz_stats_initialized.fill_(True)

    def _normalize_xyz(self, coords: torch.Tensor) -> torch.Tensor:
        """Z-score normalize coordinates."""
        return (coords - self.xyz_mean) / self.xyz_std

    @torch.no_grad()
    def _enqueue_and_dequeue(self, embeds: torch.Tensor, coords: torch.Tensor):
        """Update the memory queue with current batch (FIFO, wraparound)."""
        B = embeds.shape[0]
        Q = self.queue_size
        ptr = int(self.queue_ptr)

        embeds_norm = F.normalize(embeds.detach().float(), dim=1)
        coords_f = coords.detach().float()

        end = min(ptr + B, Q)
        first = end - ptr
        self.queue[ptr:end] = embeds_norm[:first]
        self.queue_coords[ptr:end] = coords_f[:first]

        if first < B:  # wraparound
            rem = B - first
            self.queue[:rem] = embeds_norm[first:]
            self.queue_coords[:rem] = coords_f[first:]

        self.queue_ptr[0] = (ptr + B) % Q
        self.queue_filled[0] = min(int(self.queue_filled) + B, Q)

    def _compute_dist_contrastive_loss(
        self, chunk_embeds: torch.Tensor, coords: torch.Tensor,
    ) -> torch.Tensor:
        """Soft distance-weighted contrastive loss on embeddings.

        Spatially close probes get similar embeddings, far probes get dissimilar.
        Uses Gaussian kernel over pairwise Euclidean distances as soft targets.

        Args:
            chunk_embeds: [B, D] mean-pooled student encoder output
            coords: [B, 3] raw CCF coordinates in µm

        Returns:
            Scalar loss
        """
        z = F.normalize(chunk_embeds, dim=-1)  # [B, D]
        B = z.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=z.device)

        # Extend keys with memory queue (if enabled and has entries)
        if self.queue_size > 0 and int(self.queue_filled) > 0:
            filled = int(self.queue_filled)
            queue_z = self.queue[:filled].to(z.dtype)       # [Q, D] detached buffer
            queue_c = self.queue_coords[:filled].to(z.dtype)  # [Q, 3]
            all_z = torch.cat([z, queue_z], dim=0)          # [B+Q, D]
            all_c = torch.cat([coords.float(), queue_c], dim=0)  # [B+Q, 3]
        else:
            all_z = z
            all_c = coords.float()

        K = all_z.shape[0]  # B or B+Q

        # Similarity: queries [B, D] vs keys [K, D] -> [B, K]
        sim_matrix = z @ all_z.T / self.dist_temperature

        # Mask self (diagonal in first B columns only)
        diag_mask = torch.zeros(B, K, device=z.device, dtype=torch.bool)
        diag_mask[:, :B] = torch.eye(B, device=z.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(diag_mask, float("-inf"))

        log_probs = F.log_softmax(sim_matrix, dim=-1)  # [B, K]

        # Soft targets from spatial distances between queries and all keys
        with torch.no_grad():
            diff = coords.float().unsqueeze(1) - all_c.unsqueeze(0)  # [B, K, 3]
            dist_sq = (diff ** 2).sum(dim=-1)  # [B, K]
            soft_target = torch.exp(-dist_sq / (2.0 * self.dist_sigma ** 2))
            soft_target.masked_fill_(diag_mask, 0.0)
            row_sums = soft_target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            soft_target = soft_target / row_sums  # [B, K]

        log_probs = log_probs.masked_fill(diag_mask, 0.0)
        loss = -(soft_target * log_probs).sum(dim=-1).mean()
        return loss

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
        idx_j = idx_j + (idx_j >= idx_i).long()  # shift to avoid self-pairs

        # Concatenate pair embeddings
        z_i = chunk_embeds[idx_i]  # [B, D]
        z_j = chunk_embeds[idx_j]  # [B, D]
        pair_embeds = torch.cat([z_i, z_j], dim=-1)  # [B, 2*D]

        # Predict distance
        pred_dist = self.dist_pred_head(pair_embeds).squeeze(-1)  # [B]

        # Target: log(1 + euclidean_distance)
        with torch.no_grad():
            coord_diff = coords[idx_i] - coords[idx_j]  # [B, 3]
            euclidean_dist = torch.sqrt((coord_diff ** 2).sum(dim=-1) + 1e-8)
            target_dist = torch.log1p(euclidean_dist)

        return F.mse_loss(pred_dist, target_dist)

    @staticmethod
    def _compute_vicreg_loss(z: torch.Tensor, gamma: float = 1.0):
        """VICReg variance + covariance regularization (Bardes et al. 2021).

        Args:
            z: [B, D] mean-pooled encoder embeddings (NOT l2-normalized)
            gamma: target std per dimension (default 1.0)

        Returns:
            (var_loss, cov_loss) — both scalar tensors on same device as z
        """
        N, D = z.shape
        if N < 2:
            zero = torch.tensor(0.0, device=z.device)
            return zero, zero
        z = z - z.mean(dim=0)
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        var_loss = F.relu(gamma - std).mean()
        cov = (z.T @ z) / (N - 1)
        diag_mask = torch.eye(D, device=z.device, dtype=torch.bool)
        cov_loss = cov.masked_fill(diag_mask, 0.0).pow(2).sum() / D
        return var_loss, cov_loss

    @torch.no_grad()
    def _compute_conv_targets(self, mel: torch.Tensor) -> torch.Tensor:
        """Conv-layer features as stop-grad reconstruction targets.

        Runs the two WhisperEncoder conv layers on unmasked mel to get
        pre-transformer features [B, 1500, D]. Used as targets for the
        single-pass masked reconstruction loss (no teacher needed).
        """
        conv1_out = F.gelu(self.student.conv1(mel))      # [B, D, 3000]
        conv2_out = F.gelu(self.student.conv2(conv1_out)) # [B, D, 1500]
        return conv2_out.transpose(1, 2)                  # [B, 1500, D]

    def ssl_forward(self, input_values: torch.Tensor,
                    coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """EMA teacher-student SSL forward pass.

        Each 3s chunk is padded to 30s by the feature extractor. The full
        1500-frame encoder output is used for masked prediction SSL.

        Args:
            input_values: [B, T] raw waveform (already resampled to 16kHz)
            coords: [B, 3] optional XYZ coordinates for auxiliary loss

        Returns:
            Scalar combined loss (L_ssl + xyz_lambda * L_xyz)
        """
        mel = self._compute_mel(input_values)  # [B, 80, mel_frames]
        B = mel.shape[0]

        # Skip teacher when SSL loss is not active (saves ~50% forward pass time)
        _ssl_active = not self.normalize_loss_weights or self.ssl_lambda_raw > 0

        if _ssl_active:
            # 1. Teacher forward (no grad, no masking, output all hidden states)
            with torch.no_grad():
                teacher_out = self.teacher(mel, output_hidden_states=True)
                top_k_states = teacher_out.hidden_states[-self.average_top_k_layers:]
                target = torch.stack(list(top_k_states)).mean(dim=0)  # [B, T, D]
                target = F.instance_norm(target.transpose(1, 2)).transpose(1, 2)
            _, seq_len, D = target.shape
        else:
            # Infer encoder output length from mel: conv2 stride=2
            seq_len = mel.shape[-1] // 2

        # 2. Compute mask indices (min_masks=2 prevents zero-mask edge case)
        mask_time_indices = _compute_mask_indices(
            (B, seq_len),
            mask_prob=self.mask_time_prob,
            mask_length=self.mask_time_length,
            min_masks=2,
        )
        mask_time_indices = torch.from_numpy(mask_time_indices).to(mel.device)
        mask_bool = mask_time_indices.bool()  # [B, seq_len]

        # 3. Student forward — zero mel at masked positions before encoder
        # encoder stride 2x → each encoder frame covers 2 mel frames
        mask_mel = mask_bool.repeat_interleave(2, dim=1)  # [B, mel_frames]
        mel_for_student = mel.clone()
        mel_for_student[mask_mel.unsqueeze(1).expand_as(mel_for_student)] = 0.0
        student_out = self.student(mel_for_student)
        student_hidden = student_out.last_hidden_state  # [B, seq_len, D]

        # 4-5. SSL reconstruction loss (skip if not active)
        if _ssl_active:
            predicted = self.decoder(student_hidden)  # [B, 1500, D]
            ssl_loss = F.mse_loss(predicted[mask_bool], target[mask_bool])
        else:
            ssl_loss = torch.tensor(0.0, device=mel.device)

        # 4b. Single-pass masked reconstruction (no teacher, no EMA).
        recon_loss = torch.tensor(0.0, device=mel.device)
        if self.recon_lambda > 0:
            conv_targets = self._compute_conv_targets(mel)
            predicted_recon = self.recon_decoder(student_hidden)
            recon_loss = F.mse_loss(predicted_recon[mask_bool], conv_targets[mask_bool])

        # 5b. VICReg on mean-pooled student embeddings (anti-collapse regularizer)
        vicreg_var_loss = torch.tensor(0.0, device=mel.device)
        vicreg_cov_loss = torch.tensor(0.0, device=mel.device)
        if self.vicreg_var_coeff > 0 or self.vicreg_cov_coeff > 0:
            chunk_embeds_vic = student_hidden.mean(dim=1)  # [B, D]
            vicreg_var_loss, vicreg_cov_loss = self._compute_vicreg_loss(
                chunk_embeds_vic, self.vicreg_gamma)

        # 6. Pairwise distance prediction loss (optional, config-driven)
        distpred_loss = torch.tensor(0.0, device=mel.device)
        if (self.distpred_lambda > 0 or self.distpred_lambda_raw > 0) and coords is not None:
            chunk_embeds = student_hidden.mean(dim=1)  # [B, D]
            distpred_loss = self._compute_pairwise_dist_loss(chunk_embeds, coords)

        # Compute total loss (pure SSL + optional recon/VICReg/distpred)
        loss = (ssl_loss
                + self.distpred_lambda * distpred_loss
                + self.recon_lambda * recon_loss
                + self.vicreg_var_coeff * vicreg_var_loss
                + self.vicreg_cov_coeff * vicreg_cov_loss)

        # 9. Update teacher EMA only when SSL is active
        if self.training and _ssl_active:
            self._update_ema()

        return loss

    def ssl_forward_detailed(self, input_values: torch.Tensor,
                              coords: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """EMA teacher-student SSL forward with detailed outputs for logging.

        Same computation as ssl_forward() but returns loss components and
        intermediate tensors for reconstruction visualization.

        Args:
            input_values: [B, T] raw waveform (already resampled to 16kHz)
            coords: [B, 3] optional XYZ coordinates for auxiliary loss

        Returns:
            dict with keys:
                loss: scalar combined loss (same as ssl_forward)
                ssl_loss: scalar masked reconstruction MSE
                xyz_loss: scalar coordinate prediction MSE (0.0 if xyz_lambda=0)
                mask_bool: [B, 1500] bool tensor
                predicted: [B, 1500, D] decoder output
                target: [B, 1500, D] teacher target
        """
        mel = self._compute_mel(input_values)  # [B, 80, mel_frames]
        B = mel.shape[0]

        # Skip teacher when SSL loss is not active (saves ~50% forward pass time)
        _ssl_active = not self.normalize_loss_weights or self.ssl_lambda_raw > 0

        if _ssl_active:
            # 1. Teacher forward (no grad, no masking, output all hidden states)
            with torch.no_grad():
                teacher_out = self.teacher(mel, output_hidden_states=True)
                top_k_states = teacher_out.hidden_states[-self.average_top_k_layers:]
                target = torch.stack(list(top_k_states)).mean(dim=0)  # [B, T, D]
                target = F.instance_norm(target.transpose(1, 2)).transpose(1, 2)

                # Also get student output on UNMASKED mel for teacher-student comparison
                student_unmasked_out = self.student(mel)
                student_unmasked_hidden = student_unmasked_out.last_hidden_state  # [B, T, D]
            _, seq_len, D = target.shape
        else:
            seq_len = mel.shape[-1] // 2
            target = torch.zeros(B, seq_len, self._hidden_dim, device=mel.device)
            student_unmasked_hidden = None

        # 2. Compute mask indices (min_masks=2 prevents zero-mask edge case)
        mask_time_indices = _compute_mask_indices(
            (B, seq_len),
            mask_prob=self.mask_time_prob,
            mask_length=self.mask_time_length,
            min_masks=2,
        )
        mask_time_indices = torch.from_numpy(mask_time_indices).to(mel.device)
        mask_bool = mask_time_indices.bool()  # [B, seq_len]

        # 3. Student forward — zero mel at masked positions before encoder
        mask_mel = mask_bool.repeat_interleave(2, dim=1)  # [B, mel_frames]
        mel_for_student = mel.clone()
        mel_for_student[mask_mel.unsqueeze(1).expand_as(mel_for_student)] = 0.0
        student_out = self.student(mel_for_student)
        student_hidden = student_out.last_hidden_state  # [B, seq_len, D]

        # 4-5. SSL reconstruction loss (skip decoder if not active)
        if _ssl_active:
            predicted = self.decoder(student_hidden)
            ssl_loss = F.mse_loss(predicted[mask_bool], target[mask_bool])
        else:
            predicted = torch.zeros_like(student_hidden)
            ssl_loss = torch.tensor(0.0, device=mel.device)

        # 5b. VICReg on mean-pooled student embeddings (anti-collapse regularizer)
        vicreg_var_loss = torch.tensor(0.0, device=mel.device)
        vicreg_cov_loss = torch.tensor(0.0, device=mel.device)
        if self.vicreg_var_coeff > 0 or self.vicreg_cov_coeff > 0:
            chunk_embeds_vic = student_hidden.mean(dim=1)  # [B, D]
            vicreg_var_loss, vicreg_cov_loss = self._compute_vicreg_loss(
                chunk_embeds_vic, self.vicreg_gamma)

        # 6. Pairwise distance prediction loss (optional, config-driven)
        distpred_loss = torch.tensor(0.0, device=mel.device)
        if (self.distpred_lambda > 0 or self.distpred_lambda_raw > 0) and coords is not None:
            chunk_embeds = student_hidden.mean(dim=1)  # [B, D]
            distpred_loss = self._compute_pairwise_dist_loss(chunk_embeds, coords)

        # Compute total loss (pure SSL + optional VICReg + optional distpred)
        loss = (ssl_loss
                + self.distpred_lambda * distpred_loss
                + self.vicreg_var_coeff * vicreg_var_loss
                + self.vicreg_cov_coeff * vicreg_cov_loss)

        # 7. EMA teacher-student tracking metrics
        ema_decay_current = self._get_annealed_decay()
        teacher_student_cosine = torch.tensor(0.0, device=mel.device)
        target_variance = torch.tensor(0.0, device=mel.device)
        if _ssl_active and student_unmasked_hidden is not None:
            with torch.no_grad():
                # Mean-pooled cosine between teacher target and student output (unmasked)
                t_pool = target.mean(dim=1)  # [B, D]
                s_pool = student_unmasked_hidden.mean(dim=1)  # [B, D]
                teacher_student_cosine = F.cosine_similarity(t_pool, s_pool, dim=1).mean()
                # Target variance — how informative are the teacher targets?
                target_variance = target.var(dim=1).mean()  # avg per-position variance

        # 8. Update teacher EMA only when SSL is active
        if self.training and _ssl_active:
            self._update_ema()

        return {
            "loss": loss,
            "ssl_loss": ssl_loss,
            "chunk_embeds": student_hidden.mean(dim=1),
            "xyz_loss": torch.tensor(0.0, device=mel.device),   # removed — kept for training loop compat
            "dist_loss": torch.tensor(0.0, device=mel.device),  # removed — kept for training loop compat
            "distpred_loss": distpred_loss.detach(),
            "var_loss": vicreg_var_loss.detach(),
            "mask_bool": mask_bool.detach(),
            "predicted": predicted.detach(),
            "target": target.detach(),
            # EMA teacher-student metrics
            "ema_decay": ema_decay_current,
            "teacher_student_cosine": teacher_student_cosine.detach(),
            "target_variance": target_variance.detach(),
        }

    def encode(self, input_values: torch.Tensor) -> torch.Tensor:
        """Extract mean-pooled embeddings.

        When flexible_length=True: mel is ~300 frames → encoder output ~150 frames.
        When flexible_length=False: mel is 3000 frames → encoder output 1500 frames.

        Args:
            input_values: [B, T] raw waveform (3s chunks @ 16kHz)

        Returns:
            [B, hidden_dim] per-chunk embeddings
        """
        mel = self._compute_mel(input_values)  # [B, 80, mel_frames]
        encoder_out = self.student(mel).last_hidden_state  # [B, T, D]
        return encoder_out.mean(dim=1)  # [B, D]

    def get_optimizer_params(self, lr: float) -> List[Dict[str, Any]]:
        """Return optimizer params — student + decoder + mask_emb + heads (teacher excluded)."""
        encoder_lr = lr * 0.1  # Lower LR for pretrained encoder
        groups = [
            {"params": [p for p in self.student.parameters() if p.requires_grad], "lr": encoder_lr},
            {"params": self.decoder.parameters(), "lr": lr},
            {"params": [self.mask_emb], "lr": lr},
            {"params": self.xyz_head.parameters(), "lr": lr},
        ]
        if self.distpred_lambda > 0 or self.distpred_lambda_raw > 0:
            groups.append({"params": self.dist_pred_head.parameters(), "lr": lr})
        if self.recon_lambda > 0:
            groups.append({"params": self.recon_decoder.parameters(), "lr": lr})
        return groups

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count parameters — student + decoder + mask_emb + heads (teacher excluded)."""
        distpred_p = 0
        if self.distpred_lambda > 0 or self.distpred_lambda_raw > 0:
            if trainable_only:
                distpred_p = sum(p.numel() for p in self.dist_pred_head.parameters() if p.requires_grad)
            else:
                distpred_p = sum(p.numel() for p in self.dist_pred_head.parameters())
        if trainable_only:
            student_p = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
            decoder_p = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
            xyz_p = sum(p.numel() for p in self.xyz_head.parameters() if p.requires_grad)
            mask_p = self.mask_emb.numel()
            return student_p + decoder_p + xyz_p + mask_p + distpred_p
        student_p = sum(p.numel() for p in self.student.parameters())
        decoder_p = sum(p.numel() for p in self.decoder.parameters())
        xyz_p = sum(p.numel() for p in self.xyz_head.parameters())
        mask_p = self.mask_emb.numel()
        return student_p + decoder_p + xyz_p + mask_p + distpred_p
