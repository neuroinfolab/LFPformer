"""Abstract backbone interface for SSL pretraining + classification."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

import torch
import torch.nn as nn


class BackboneBase(nn.Module, ABC):
    """Abstract backbone that all models must implement.

    The interface separates two modes:
    1. SSL pretraining: ssl_forward(x) → scalar loss
    2. Embedding extraction: encode(x) → [B, hidden_dim] for downstream classifier
    """

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        """Dimension of output embeddings (e.g., 768 for wav2vec2/WavLM)."""

    @property
    @abstractmethod
    def needs_resampling(self) -> bool:
        """True if model expects 16kHz input (wav2vec2, WavLM).
        False for native 1250 Hz models (PatchTST, MOMENT, TTM)."""

    @property
    def needs_normalization(self) -> bool:
        """True if model needs input amplitude scaling (e.g., Whisper uses a
        fixed log-mel spectrogram where LFP amplitudes ~0.0001 overlap with
        the silence floor). When True, preprocess_batch() applies a global
        scale factor (×3000) to bring signals into audio-like range while
        preserving cross-sample amplitude differences.
        False for models with learned feature extractors (wav2vec2, WavLM)."""
        return False

    @property
    def input_sr(self) -> int:
        """Expected input sample rate after preprocessing."""
        return 16000 if self.needs_resampling else 1250

    @abstractmethod
    def ssl_forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """SSL pretraining forward pass.

        Args:
            input_values: [B, T] raw waveform (already resampled if needed)

        Returns:
            Scalar loss tensor for backpropagation
        """

    @abstractmethod
    def encode(self, input_values: torch.Tensor) -> torch.Tensor:
        """Extract embeddings for fine-tuning/evaluation.

        Args:
            input_values: [B, T] raw waveform

        Returns:
            [B, hidden_dim] mean-pooled representation
        """

    def forward(self, input_values: torch.Tensor, mode: str = "ssl_forward", **kwargs):
        """Unified dispatch entry-point so DDP grad sync works.

        DistributedDataParallel only installs autograd hooks via the wrapped
        module's ``forward()``. Calling ``backbone.ssl_forward(x)`` directly on
        a DDP-wrapped model skips those hooks and silently breaks cross-rank
        gradient sync. Routing every training-time forward through this method
        (and calling ``backbone(input_values, mode=...)``) restores correct DDP
        behavior. Single-GPU paths are unaffected.

        Modes:
            - "ssl_forward": calls self.ssl_forward(input_values, **kwargs)
            - "ssl_forward_detailed": calls self.ssl_forward_detailed(input_values, **kwargs)
            - "encode": calls self.encode(input_values)
        """
        if mode == "ssl_forward":
            return self.ssl_forward(input_values, **kwargs)
        if mode == "ssl_forward_detailed":
            return self.ssl_forward_detailed(input_values, **kwargs)
        if mode == "encode":
            return self.encode(input_values)
        raise ValueError(f"Unknown forward mode: {mode}")

    def get_optimizer_params(self, lr: float) -> List[Dict[str, Any]]:
        """Return optimizer parameter groups. Override for per-layer LR."""
        return [{"params": self.parameters(), "lr": lr}]

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class ClassifierHead(nn.Module):
    """Shared classification head used by all backbones during fine-tuning.

    Architecture matches the paper: mean-pool encoder output → linear → 13 classes.
    """

    def __init__(self, backbone: BackboneBase, num_classes: int = 13,
                 freeze_encoder: bool = False):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.hidden_dim, num_classes)
        self.freeze_encoder = freeze_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                emb = self.backbone.encode(x)
        else:
            emb = self.backbone.encode(x)
        return self.classifier(emb)
