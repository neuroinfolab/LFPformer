"""PatchTST backbone — patch-based transformer for time series (native 1250 Hz)."""

import torch
import torch.nn as nn

from transformers import PatchTSTConfig, PatchTSTForPretraining, PatchTSTModel

from backbones.base import BackboneBase


class PatchTSTBackbone(BackboneBase):
    """PatchTST backbone for SSL pretraining on LFP signals.

    Key advantages over wav2vec2:
    - Works at native 1250 Hz (no resampling needed, no information loss)
    - Patch-based tokenization designed for time series
    - Built-in masked patch pretraining
    - Channel-independent processing
    """

    def __init__(self, config: dict):
        super().__init__()

        self.context_length = config.get("context_length", 3750)  # 3s at 1250 Hz
        self.patch_length = config.get("patch_length", 64)
        self.patch_stride = config.get("patch_stride", 32)
        self._hidden_dim = config.get("d_model", 256)

        patchtst_config = PatchTSTConfig(
            num_input_channels=1,  # single-channel LFP
            context_length=self.context_length,
            patch_length=self.patch_length,
            patch_stride=self.patch_stride,
            d_model=self._hidden_dim,
            num_attention_heads=config.get("num_heads", 8),
            num_hidden_layers=config.get("num_layers", 6),
            ffn_dim=config.get("ffn_dim", self._hidden_dim * 4),
            dropout=config.get("dropout", 0.1),
            attention_dropout=config.get("attention_dropout", 0.1),
            # Masked pretraining config
            mask_type="random",
            random_mask_ratio=config.get("mask_ratio", 0.4),
            mask_value=0.0,
            # Normalization
            scaling=config.get("scaling", "std"),  # per-sample standardization
        )

        pretrained_id = config.get("pretrained_model", None)
        rand_init = config.get("rand_init", True)  # default: train from scratch

        if rand_init or pretrained_id is None:
            self.ssl_model = PatchTSTForPretraining(patchtst_config)
            self.encoder = self.ssl_model.model
        else:
            self.ssl_model = PatchTSTForPretraining.from_pretrained(
                pretrained_id, config=patchtst_config
            )
            self.encoder = self.ssl_model.model

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def needs_resampling(self) -> bool:
        return False

    def _reshape_input(self, input_values: torch.Tensor) -> torch.Tensor:
        """Reshape [B, T] → [B, T, 1] for PatchTST (batch, seq_length, num_channels)."""
        if input_values.dim() == 2:
            return input_values.unsqueeze(-1)  # [B, T, 1]
        return input_values

    def ssl_forward(self, input_values: torch.Tensor) -> torch.Tensor:
        x = self._reshape_input(input_values)  # [B, T, 1]
        outputs = self.ssl_model(past_values=x)
        return outputs.loss

    def encode(self, input_values: torch.Tensor) -> torch.Tensor:
        x = self._reshape_input(input_values)  # [B, T, 1]
        outputs = self.encoder(past_values=x)
        # last_hidden_state: [B, num_channels, num_patches, d_model]
        hidden = outputs.last_hidden_state
        # Mean-pool over channels and patches → [B, d_model]
        return hidden.mean(dim=(1, 2))
