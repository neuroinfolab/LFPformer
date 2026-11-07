"""Wav2Vec2 backbone — contrastive SSL with optional XYZ/dist_contrastive auxiliary loss.

Wraps HuggingFace Wav2Vec2ForPreTraining with the contrastive masked
prediction objective from the Lfp2vec paper. Optionally adds:
  - XYZ coordinate prediction head (anti-collapse regularizer)
  - Soft distance-weighted contrastive loss (spatially ground embeddings)

L_total = L_contrastive + xyz_lambda * L_xyz + dist_lambda * L_dist + distpred_lambda * L_distpred
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers import Wav2Vec2Config, Wav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices,
    _sample_negative_indices,
)

from backbones.base import BackboneBase


def _sample_negative_indices_cross_sample(
    features_shape: tuple,
    num_negatives: int,
    mask_time_indices: "np.ndarray | None" = None,
) -> np.ndarray:
    """Sample negatives from ALL masked positions across the entire batch.

    Unlike HuggingFace's _sample_negative_indices which samples negatives only
    from the same utterance (sample), this function samples from the full batch.

    For LFP signals with short sequences (3 sec → 150 time steps → ~12 masked
    positions), within-sample negatives are severely limited (~11 unique).
    Cross-sample negatives give access to batch_size × masked_positions unique
    candidates (e.g., 32 × 12 = 384).

    Args:
        features_shape: (batch_size, sequence_length)
        num_negatives: number of negative samples per masked position
        mask_time_indices: binary mask [B, T] indicating which positions are masked

    Returns:
        sampled_negative_indices: [B, T, num_negatives] flat indices into [B*T]
    """
    batch_size, sequence_length = features_shape

    if mask_time_indices is None:
        mask_time_indices = np.ones(features_shape, dtype=bool)
    else:
        mask_time_indices = mask_time_indices.astype(bool)

    # Collect ALL masked positions across the batch as flat indices
    # flat_index = batch_idx * sequence_length + time_idx
    all_masked_flat = []
    for b in range(batch_size):
        masked_positions = np.where(mask_time_indices[b])[0]
        all_masked_flat.extend(b * sequence_length + masked_positions)
    all_masked_flat = np.array(all_masked_flat)
    total_masked = len(all_masked_flat)

    sampled_negative_indices = np.zeros(
        (batch_size, sequence_length, num_negatives), dtype=np.int32
    )

    for batch_idx in range(batch_size):
        masked_positions = np.where(mask_time_indices[batch_idx])[0]
        n_masked = len(masked_positions)
        if n_masked == 0:
            continue

        for local_idx, t in enumerate(masked_positions):
            # The positive's flat index (must be excluded from negatives)
            pos_flat = batch_idx * sequence_length + t

            # Sample from all masked positions except the positive
            # Build candidate pool excluding the positive
            candidates = all_masked_flat[all_masked_flat != pos_flat]
            n_candidates = len(candidates)

            if n_candidates == 0:
                continue

            # Sample with replacement if num_negatives > n_candidates
            neg_indices = np.random.randint(0, n_candidates, size=num_negatives)
            sampled_negative_indices[batch_idx, t] = candidates[neg_indices]

    return sampled_negative_indices


class Wav2Vec2Backbone(BackboneBase):
    """Wav2Vec2 backbone for SSL pretraining on LFP signals.

    Wraps HuggingFace Wav2Vec2ForPreTraining with the contrastive masked
    prediction objective from the Lfp2vec paper.

    Optional XYZ auxiliary loss forces spatially-grounded representations
    by predicting electrode CCF coordinates from mean-pooled embeddings.
    """

    def __init__(self, config: dict):
        super().__init__()

        self._hidden_dim = config.get("hidden_size", 768)
        # Default True = LFP path (1250Hz stored -> resampled to 16kHz at train). Set False
        # for data already stored at 16kHz (e.g. the raw-AP-waveform compact) so it is NOT
        # re-resampled as if it were 1250Hz.
        self._needs_resampling = bool(config.get("needs_resampling", True))

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
            layer_norm_eps=1e-12,
            feat_extract_norm="group",
            feat_proj_dropout=0.0,
            feat_extract_activation="gelu",
            feat_quantizer_dropout=0.0,
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
            contrastive_logits_temperature=float(config.get("contrastive_temperature", 0.1)),
            num_negatives=config.get("num_negatives", 100),
            # 256 = wav2vec2-base; facebook/wav2vec2-large needs 768 (added 10-12 for the large gate)
            codevector_dim=int(config.get("codevector_dim", 256)),
            proj_codevector_dim=int(config.get("proj_codevector_dim", 256)),
            diversity_loss_weight=0.1,
            ctc_loss_reduction="sum",
            ctc_zero_infinity=False,
            use_weighted_layer_sum=False,
            classifier_proj_size=256,
        )

        pretrained_id = config.get("pretrained_model", "facebook/wav2vec2-base")
        rand_init = config.get("rand_init", False)

        if rand_init or pretrained_id is None:
            self.model = Wav2Vec2ForPreTraining(w2v2_config)
        else:
            self.model = Wav2Vec2ForPreTraining.from_pretrained(
                pretrained_id, config=w2v2_config
            )

        self._cached_seq_len: Optional[int] = None

        # Optionally freeze CNN feature extractor (keep pretrained speech features)
        if config.get("freeze_feature_encoder", False):
            self.model.freeze_feature_encoder()

        # Partial encoder freezing: freeze all but top-k transformer layers.
        #   unfreeze_top_k_layers=num_hidden_layers (default) → fully unfrozen
        #   unfreeze_top_k_layers=0 → whole encoder frozen; only aux heads train
        # Required for stable CD-only training: with ssl_lambda=0 and the whole
        # encoder unfrozen, the vector quantizer codebook collapses to std=0 by
        # epoch ~3. Freezing the lower layers (and codebook) keeps the
        # pretrained speech features intact and lets only the top layers learn
        # from the CD signal.
        self._num_encoder_layers = int(config.get("num_hidden_layers", 12))
        unfreeze_k = int(config.get("unfreeze_top_k_layers",
                                    self._num_encoder_layers))
        unfreeze_k = max(0, min(unfreeze_k, self._num_encoder_layers))
        if unfreeze_k < self._num_encoder_layers:
            self.model.wav2vec2.encoder.requires_grad_(False)
            if unfreeze_k > 0:
                for layer in self.model.wav2vec2.encoder.layers[-unfreeze_k:]:
                    layer.requires_grad_(True)
                self.model.wav2vec2.encoder.layer_norm.requires_grad_(True)
            print(f"[wav2vec2] Partial freeze: top {unfreeze_k}/"
                  f"{self._num_encoder_layers} encoder layers unfrozen")

        # Gumbel-Softmax temperature annealing (wav2vec2 paper: 2.0 → 0.5)
        # HuggingFace does NOT implement this — we must do it manually.
        self.gumbel_temp_init = float(config.get("gumbel_temp_init", 2.0))
        self.gumbel_temp_min = float(config.get("gumbel_temp_min", 0.5))
        self.gumbel_temp_decay = float(config.get("gumbel_temp_decay", 0.999995))
        # Set initial temperature on the quantizer
        self.model.quantizer.temperature = self.gumbel_temp_init

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
        # Adaptive weighting (matches train_ssl_channel_dist.py's CD recipe): when
        # dist_target_fraction > 0, the dist loss is rescaled EACH STEP so its
        # contribution equals dist_target_fraction × the SSL-loss magnitude, instead
        # of the static dist_lambda. This is what lets CCFDistc (this backbone path)
        # be weighted identically to ChanDistc (which uses the same 1.0 target). 0 =
        # off = original static-dist_lambda behaviour (fully backward-compatible).
        self.dist_target_fraction = float(config.get("dist_target_fraction", 0.0))
        self._last_dist_lambda_eff = 0.0   # populated each forward for logging
        self.dist_temperature = float(config.get("dist_temperature", 0.2))
        self.dist_sigma = float(config.get("dist_sigma", 500.0))  # µm
        # Channel distance mode: use hardware channel geometry instead of CCF coords
        # When true, coords is [B] channel numbers; distance = probe shaft geometry
        # samples_per_probe: loss computed within each probe group (0 = whole batch)
        self.channel_distance = bool(config.get("channel_distance", False))
        self.samples_per_probe = int(config.get("samples_per_probe", 0))

        # VICReg regularization (prevents cosine similarity collapse)
        self.vicreg_var_lambda = float(config.get("vicreg_var_lambda", 0.0))
        self.vicreg_cov_lambda = float(config.get("vicreg_cov_lambda", 0.0))
        self.vicreg_target_std = float(config.get("vicreg_target_std", 1.0))

        # Bypass quantizer mode: use raw CNN features as contrastive targets
        # instead of Gumbel-Softmax quantized codes
        self.bypass_quantizer = bool(config.get("bypass_quantizer", False))

        # Cross-sample negative sampling: sample negatives from other samples
        # in the batch instead of only from the same sample.
        # Addresses soft cosine collapse for short LFP sequences where
        # within-sample negatives are limited to ~11 unique vectors.
        self.cross_sample_negatives = bool(config.get("cross_sample_negatives", False))

        # Pairwise distance prediction loss (regression MLP)
        self.distpred_lambda = float(config.get("distpred_lambda", 0.0))
        if self.distpred_lambda > 0:
            D = self._hidden_dim
            self.dist_pred_head = nn.Sequential(
                nn.Linear(2 * D, 256),
                nn.GELU(),
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def needs_resampling(self) -> bool:
        return self._needs_resampling

    @property
    def encoder(self):
        """Alias for wav2vec2 encoder (compatibility with finetune code)."""
        return self.model.wav2vec2.encoder

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

    def step_gumbel_temperature(self):
        """Decay Gumbel-Softmax temperature by one step. Call after each optimizer step."""
        current = self.model.quantizer.temperature
        new_temp = max(current * self.gumbel_temp_decay, self.gumbel_temp_min)
        self.model.quantizer.temperature = new_temp

    @property
    def gumbel_temperature(self) -> float:
        return self.model.quantizer.temperature

    def _compute_masks(
        self, input_values: torch.Tensor
    ) -> tuple:
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
            min_masks=self.model.config.mask_time_min_masks,
        )

        if self.cross_sample_negatives:
            sampled_negatives = _sample_negative_indices_cross_sample(
                (batch_size, seq_len),
                num_negatives=self.model.config.num_negatives,
                mask_time_indices=mask_time_indices,
            )
        else:
            sampled_negatives = _sample_negative_indices(
                (batch_size, seq_len),
                num_negatives=self.model.config.num_negatives,
                mask_time_indices=mask_time_indices,
            )

        mask_time_indices = torch.from_numpy(mask_time_indices).to(device)
        sampled_negatives = torch.from_numpy(sampled_negatives).to(device)

        return mask_time_indices, sampled_negatives

    def _compute_dist_contrastive_single_group(
        self, z: torch.Tensor, dist_sq: torch.Tensor,
    ) -> torch.Tensor:
        """KL-divergence loss for one group of samples.

        Args:
            z: [G, D] normalized embeddings
            dist_sq: [G, G] pairwise squared distances in µm²
        """
        G = z.shape[0]
        if G < 2:
            return torch.tensor(0.0, device=z.device)

        sim_matrix = z @ z.T / self.dist_temperature
        diag_mask = torch.eye(G, device=z.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(diag_mask, float("-inf"))
        log_probs = F.log_softmax(sim_matrix, dim=-1)

        with torch.no_grad():
            soft_target = torch.exp(-dist_sq / (2.0 * self.dist_sigma ** 2))
            soft_target.masked_fill_(diag_mask, 0.0)
            row_sums = soft_target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            soft_target = soft_target / row_sums

        log_probs = log_probs.masked_fill(diag_mask, 0.0)
        return -(soft_target * log_probs).sum(dim=-1).mean()

    def _compute_dist_contrastive_loss(
        self, chunk_embeds: torch.Tensor, coords: torch.Tensor,
    ) -> torch.Tensor:
        """Soft distance-weighted contrastive loss on embeddings.

        Two modes:
        - CCF mode (channel_distance=False): coords is [B, 3] CCF coordinates,
          distance is Euclidean in 3D space. All pairs in batch contribute.
        - Channel mode (channel_distance=True): coords is [B] channel numbers,
          distance is NP1.0 4-column geometry (x={11,27,43,59}µm, 20µm row pitch).
          Loss computed within each probe group (samples_per_probe) and averaged.
        """
        z = F.normalize(chunk_embeds, dim=-1)
        B = z.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=z.device)

        if self.channel_distance:
            # NP1.0 4-column geometry: coords is [B] channel numbers
            ch = coords.long()
            rows = ch // 2
            col_in_row = ch % 2
            even_row = (rows % 2 == 0)
            x = torch.where(
                even_row,
                torch.where(col_in_row == 0,
                             torch.tensor(11.0, device=ch.device),
                             torch.tensor(43.0, device=ch.device)),
                torch.where(col_in_row == 0,
                             torch.tensor(27.0, device=ch.device),
                             torch.tensor(59.0, device=ch.device)),
            ).float()
            y = rows.float() * 20.0
            dx = x.unsqueeze(0) - x.unsqueeze(1)
            dy = y.unsqueeze(0) - y.unsqueeze(1)
            dist_sq = dx ** 2 + dy ** 2  # [B, B]

            # Compute within probe groups
            spp = self.samples_per_probe
            if spp > 0 and spp < B:
                num_groups = B // spp
                losses = []
                for g in range(num_groups):
                    s, e = g * spp, (g + 1) * spp
                    losses.append(self._compute_dist_contrastive_single_group(
                        z[s:e], dist_sq[s:e, s:e]))
                return torch.stack(losses).mean()
            else:
                return self._compute_dist_contrastive_single_group(z, dist_sq)
        else:
            # CCF mode: coords is [B, 3]
            with torch.no_grad():
                diff = coords.unsqueeze(0) - coords.unsqueeze(1)
                dist_sq = (diff ** 2).sum(dim=-1)
            return self._compute_dist_contrastive_single_group(z, dist_sq)

    def _compute_pairwise_dist_loss(
        self, chunk_embeds: torch.Tensor, coords: torch.Tensor,
    ) -> torch.Tensor:
        """Pairwise distance prediction loss.

        Samples B random pairs from the batch, concatenates their embeddings,
        and regresses the log-scaled Euclidean distance between CCF coordinates.
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

    def _compute_vicreg_loss(self, embeddings: torch.Tensor) -> tuple:
        """Compute VICReg variance + covariance regularization on embeddings.

        Args:
            embeddings: [B, D] mean-pooled encoder output

        Returns:
            (var_loss, cov_loss) tuple
        """
        B, D = embeddings.shape
        if B < 2:
            zero = torch.tensor(0.0, device=embeddings.device)
            return zero, zero

        # Variance loss: penalize dimensions with std below target
        # ReLU(target - std) → only penalizes low-variance dims
        std = embeddings.std(dim=0)  # [D]
        var_loss = F.relu(self.vicreg_target_std - std).mean()

        # Covariance loss: penalize off-diagonal correlations
        # Encourages each dimension to encode independent information
        emb_centered = embeddings - embeddings.mean(dim=0)
        cov = (emb_centered.T @ emb_centered) / (B - 1)  # [D, D]
        # Zero out diagonal (we only penalize off-diagonal)
        cov.fill_diagonal_(0.0)
        cov_loss = (cov ** 2).sum() / D

        return var_loss, cov_loss

    def ssl_forward_detailed(self, input_values: torch.Tensor,
                             coords: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Contrastive SSL forward with separate loss components and perplexity.

        Supports two modes:
        - Standard (bypass_quantizer=False): wav2vec2 contrastive with Gumbel quantizer
        - Bypass (bypass_quantizer=True): contrastive on raw features, no quantizer

        Both modes support optional VICReg regularization on encoder output.

        Returns dict with:
            loss: scalar total loss
            ssl_loss: contrastive + diversity (combined HF loss)
            contrastive_loss: contrastive component only
            diversity_loss: diversity component only
            codevector_perplexity: codebook utilization (healthy: 100-500)
            gumbel_temperature: current Gumbel-Softmax temperature
            var_loss, cov_loss: VICReg regularization losses
            xyz_loss, dist_loss, distpred_loss: auxiliary losses
        """
        mask_time_indices, sampled_negatives = self._compute_masks(input_values)

        if self.bypass_quantizer:
            # Bypass mode: run encoder without quantizer, use raw features for contrastive
            encoder_out = self.model.wav2vec2(input_values)
            hidden_states = encoder_out.last_hidden_state  # [B, seq_len, D=768]

            # Extract features (pre-transformer) for contrastive targets
            # After feature_projection, features are [B, seq_len, D=768] (same as hidden_states)
            extract_features = self.model.wav2vec2.feature_extractor(input_values)
            extract_features = extract_features.transpose(1, 2)
            extract_features = self.model.wav2vec2.feature_projection(extract_features)[0]

            # Both context and targets are 768-dim → use project_hid for both
            # (project_q expects 256-dim quantized codes, can't be used here)
            context = self.model.project_hid(hidden_states)  # [B, T, proj_dim=256]
            targets = self.model.project_hid(extract_features.detach())  # [B, T, proj_dim=256]

            # Contrastive loss at masked positions only
            mask_bool = mask_time_indices.bool()
            n_masked = mask_bool.sum()

            if n_masked > 0:
                context_masked = context[mask_bool]  # [N, proj_dim]
                targets_masked = targets[mask_bool]  # [N, proj_dim]

                # Cosine similarity / temperature
                context_norm = F.normalize(context_masked, dim=-1)
                targets_norm = F.normalize(targets_masked, dim=-1)

                # Sample negatives from targets
                neg_indices = sampled_negatives[mask_bool.cpu()].to(input_values.device)  # [N, num_neg]
                # Gather negative targets
                targets_flat = targets.reshape(-1, targets.shape[-1])  # [B*T, proj_dim]
                neg_targets = targets_flat[neg_indices.reshape(-1)].reshape(
                    neg_indices.shape[0], neg_indices.shape[1], -1)  # [N, num_neg, proj_dim]
                neg_targets_norm = F.normalize(neg_targets, dim=-1)

                # Logits: [N, 1+num_neg]
                pos_sim = (context_norm * targets_norm).sum(dim=-1, keepdim=True) / 0.1  # temp=0.1
                neg_sim = torch.bmm(neg_targets_norm, context_norm.unsqueeze(-1)).squeeze(-1) / 0.1
                logits = torch.cat([pos_sim, neg_sim], dim=-1)

                contrastive_loss = F.cross_entropy(logits, torch.zeros(logits.shape[0],
                    dtype=torch.long, device=logits.device), reduction="sum")
            else:
                contrastive_loss = torch.tensor(0.0, device=input_values.device)

            diversity_loss = torch.tensor(0.0, device=input_values.device)
            codevector_perplexity = torch.tensor(0.0, device=input_values.device)
            ssl_loss = contrastive_loss

            # Mean-pooled embeddings for auxiliary losses + VICReg
            chunk_embeds = hidden_states.mean(dim=1)
        else:
            # Standard mode: wav2vec2 with quantizer
            # Request hidden states if we need encoder embeddings for auxiliary losses
            # Also extract if channel_distance mode is on (external loss computation)
            needs_embeds = (self.vicreg_var_lambda > 0 or self.vicreg_cov_lambda > 0
                            or self.xyz_lambda > 0 or self.dist_lambda > 0
                            or self.distpred_lambda > 0
                            or self.channel_distance)
            outputs = self.model(
                input_values=input_values,
                mask_time_indices=mask_time_indices,
                sampled_negative_indices=sampled_negatives,
                output_hidden_states=needs_embeds,
            )

            contrastive_loss = outputs.contrastive_loss
            diversity_loss = outputs.diversity_loss
            codevector_perplexity = outputs.codevector_perplexity
            ssl_loss = outputs.loss

            if needs_embeds:
                chunk_embeds = outputs.hidden_states[-1].mean(dim=1)
            else:
                chunk_embeds = None

        # VICReg regularization
        var_loss = torch.tensor(0.0, device=input_values.device)
        cov_loss = torch.tensor(0.0, device=input_values.device)
        if (self.vicreg_var_lambda > 0 or self.vicreg_cov_lambda > 0):
            if chunk_embeds is None:
                encoder_out = self.model.wav2vec2(input_values)
                chunk_embeds = encoder_out.last_hidden_state.mean(dim=1)
            var_loss, cov_loss = self._compute_vicreg_loss(chunk_embeds)

        # Auxiliary losses
        xyz_loss = torch.tensor(0.0, device=input_values.device)
        dist_loss = torch.tensor(0.0, device=input_values.device)
        distpred_loss = torch.tensor(0.0, device=input_values.device)

        needs_aux = (self.xyz_lambda > 0 or self.dist_lambda > 0
                     or self.distpred_lambda > 0) and coords is not None
        if needs_aux:
            if chunk_embeds is None:
                encoder_out = self.model.wav2vec2(input_values)
                chunk_embeds = encoder_out.last_hidden_state.mean(dim=1)

            if self.xyz_lambda > 0:
                if not self.xyz_stats_initialized:
                    self._init_xyz_stats(coords)
                pred_xyz = self.xyz_head(chunk_embeds)
                target_xyz = self._normalize_xyz(coords)
                xyz_loss = F.mse_loss(pred_xyz, target_xyz)

            if self.dist_lambda > 0:
                dist_loss = self._compute_dist_contrastive_loss(chunk_embeds, coords)

            if self.distpred_lambda > 0:
                distpred_loss = self._compute_pairwise_dist_loss(chunk_embeds, coords)

        # Effective dist coefficient: adaptive (target_fraction × ssl/dist, detached
        # so it's a per-step scalar not a backprop path) when dist_target_fraction>0,
        # else the static dist_lambda. Guard the division against a ~0 dist_loss.
        dist_coef = self.dist_lambda
        if self.dist_target_fraction > 0 and self.dist_lambda > 0:
            _dl = float(dist_loss.detach())
            if _dl > 1e-8:
                _sl = float(ssl_loss.detach())
                dist_coef = self.dist_target_fraction * _sl / _dl
        self._last_dist_lambda_eff = float(dist_coef)

        loss = (ssl_loss
                + self.vicreg_var_lambda * var_loss
                + self.vicreg_cov_lambda * cov_loss
                + self.xyz_lambda * xyz_loss
                + dist_coef * dist_loss
                + self.distpred_lambda * distpred_loss)

        return {
            "loss": loss,
            "ssl_loss": ssl_loss,
            "contrastive_loss": contrastive_loss,
            "diversity_loss": diversity_loss,
            "codevector_perplexity": codevector_perplexity,
            "gumbel_temperature": self.model.quantizer.temperature,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
            "xyz_loss": xyz_loss,
            "dist_loss": dist_loss,
            "dist_lambda_eff": self._last_dist_lambda_eff,
            "distpred_loss": distpred_loss,
            "chunk_embeds": chunk_embeds,
        }

    def ssl_forward(self, input_values: torch.Tensor,
                    coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Contrastive SSL forward pass. Delegates to ssl_forward_detailed for simplicity."""
        result = self.ssl_forward_detailed(input_values, coords=coords)
        return result["loss"]

    def encode(self, input_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state  # [B, seq_len, hidden_dim]
        return hidden_states.mean(dim=1)  # [B, hidden_dim]

    def encode_full(self, input_values: torch.Tensor) -> torch.Tensor:
        """Per-frame transformer features (no mean-pool). [B, T_frames, hidden_dim].

        Used by the LFP-reconstruction decoder — mean-pooling discards the
        temporal information needed to reproduce 3750 individual samples."""
        outputs = self.model.wav2vec2(input_values)
        return outputs.last_hidden_state

    def get_optimizer_params(self, lr: float) -> List[Dict[str, Any]]:
        """Return optimizer params — model + all aux heads with separate LR groups."""
        has_aux = self.xyz_lambda > 0 or self.distpred_lambda > 0
        if has_aux:
            model_params = [p for p in self.model.parameters() if p.requires_grad]
            groups = [{"params": model_params, "lr": lr}]
            # Always include xyz_head (it exists unconditionally)
            groups.append({"params": list(self.xyz_head.parameters()), "lr": lr})
            if self.distpred_lambda > 0:
                groups.append({"params": list(self.dist_pred_head.parameters()), "lr": lr})
            return groups
        return [{"params": self.parameters(), "lr": lr}]

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count parameters including XYZ head."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
