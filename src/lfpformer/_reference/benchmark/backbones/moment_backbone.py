"""MOMENT backbone — task-agnostic time series foundation model (native 1250 Hz)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.base import BackboneBase


class MOMENTBackbone(BackboneBase):
    """MOMENT backbone for SSL pretraining on LFP signals.

    Key advantages:
    - Pre-trained on diverse time series collection ("Time-series Pile")
    - Task-agnostic: supports forecasting, classification, anomaly detection
    - Works at native 1250 Hz (no resampling)
    - Strong zero-shot and few-shot performance

    Uses momentfm package: pip install momentfm
    """

    def __init__(self, config: dict):
        super().__init__()

        from momentfm import MOMENTPipeline

        self._hidden_dim = config.get("d_model", 1024)
        self.seq_length = config.get("context_length", 3750)

        pretrained_id = config.get(
            "pretrained_model", "AutonLab/MOMENT-1-large"
        )
        rand_init = config.get("rand_init", False)

        if rand_init or pretrained_id is None:
            # MOMENT doesn't easily support random init — use smallest pretrained
            pretrained_id = "AutonLab/MOMENT-1-small"

        # Load for reconstruction (SSL-like masked pretraining)
        # MOMENTPipeline IS the model (nn.Module) — no .model attribute
        self.moment_model = MOMENTPipeline.from_pretrained(
            pretrained_id,
            model_kwargs={
                "task_name": "reconstruction",
                "seq_len": self.seq_length,
                "forecast_horizon": 0,
            },
        )
        self.moment_model.init()

        # Override hidden_dim from actual model config
        if hasattr(self.moment_model, "config") and hasattr(self.moment_model.config, "d_model"):
            self._hidden_dim = self.moment_model.config.d_model

        # Reconstruction loss for SSL
        self.mask_ratio = config.get("mask_ratio", 0.3)

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def needs_resampling(self) -> bool:
        return False

    def _reshape_input(self, input_values: torch.Tensor) -> torch.Tensor:
        """Reshape [B, T] → [B, 1, T] for MOMENT (n_channels=1)."""
        if input_values.dim() == 2:
            return input_values.unsqueeze(1)
        return input_values

    def ssl_forward(self, input_values: torch.Tensor) -> torch.Tensor:
        x = self._reshape_input(input_values)  # [B, 1, T]

        batch_size = x.shape[0]
        seq_len = x.shape[2]
        input_mask = torch.ones(batch_size, seq_len, device=x.device)

        # MOMENT uses keyword-only args: x_enc=, input_mask=
        outputs = self.moment_model.reconstruct(x_enc=x, input_mask=input_mask)

        # Reconstruction may be truncated to patch-aligned length
        recon = outputs.reconstruction  # [B, 1, T_aligned]
        t_recon = recon.shape[2]
        loss = F.mse_loss(recon, x[:, :, :t_recon])
        return loss

    def encode(self, input_values: torch.Tensor) -> torch.Tensor:
        x = self._reshape_input(input_values)  # [B, 1, T]

        batch_size = x.shape[0]
        seq_len = x.shape[2]
        input_mask = torch.ones(batch_size, seq_len, device=x.device)

        # embed() returns [B, d_model] with reduction='mean'
        outputs = self.moment_model.embed(x_enc=x, input_mask=input_mask)
        return outputs.embeddings  # [B, d_model]
