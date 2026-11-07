"""PatchTSMixer backbone — lightweight MLP-Mixer for time series (native 1250 Hz).

Uses PatchTSMixer (available in HuggingFace transformers) instead of TinyTimeMixer.
Same role: lightweight, non-attention mixer architecture for time series SSL.
"""

import torch

from transformers import (
    PatchTSMixerConfig,
    PatchTSMixerForPretraining,
)

from backbones.base import BackboneBase


class TTMBackbone(BackboneBase):
    """PatchTSMixer backbone (MLP-Mixer) for SSL pretraining on LFP signals.

    Key advantages over attention-based models:
    - MLP-Mixer: no attention mechanism (faster, lower memory)
    - Built-in masked pretraining via PatchTSMixerForPretraining
    - Lightweight and efficient
    - Works at native 1250 Hz (no resampling)
    - Patch-based input with inter-channel and intra-channel mixing
    """

    def __init__(self, config: dict):
        super().__init__()

        self.context_length = config.get("context_length", 3750)
        self.patch_length = config.get("patch_length", 64)
        self._hidden_dim = config.get("d_model", 256)
        self.forecast_horizon = config.get("forecast_horizon", 64)

        patchtsmixer_config = PatchTSMixerConfig(
            context_length=self.context_length,
            patch_length=self.patch_length,
            num_input_channels=1,
            prediction_length=self.forecast_horizon,
            d_model=self._hidden_dim,
            num_layers=config.get("num_layers", 4),
            dropout=config.get("dropout", 0.1),
            expansion_factor=config.get("expansion_factor", 2),
            # Pretraining: masked patch reconstruction
            mask_type="random",
            random_mask_ratio=config.get("mask_ratio", 0.5),
            head_dropout=config.get("head_dropout", 0.1),
            scaling=config.get("scaling", "std"),
        )

        pretrained_id = config.get("pretrained_model", None)
        rand_init = config.get("rand_init", True)

        if rand_init or pretrained_id is None:
            self.ssl_model = PatchTSMixerForPretraining(patchtsmixer_config)
            self.encoder = self.ssl_model.model
        else:
            self.ssl_model = PatchTSMixerForPretraining.from_pretrained(
                pretrained_id, config=patchtsmixer_config
            )
            self.encoder = self.ssl_model.model

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def needs_resampling(self) -> bool:
        return False

    def _reshape_input(self, input_values: torch.Tensor) -> torch.Tensor:
        """Reshape [B, T] → [B, T, 1] for PatchTSMixer (batch, seq, channels)."""
        if input_values.dim() == 2:
            return input_values.unsqueeze(-1)
        return input_values

    def ssl_forward(self, input_values: torch.Tensor) -> torch.Tensor:
        x = self._reshape_input(input_values)  # [B, T, 1]
        outputs = self.ssl_model(past_values=x)
        return outputs.loss

    def encode(self, input_values: torch.Tensor) -> torch.Tensor:
        x = self._reshape_input(input_values)  # [B, T, 1]
        outputs = self.encoder(past_values=x)
        hidden = outputs.last_hidden_state  # [B, num_channels, num_patches, d_model]
        return hidden.mean(dim=(1, 2))  # [B, d_model]
