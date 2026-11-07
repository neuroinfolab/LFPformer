"""Whisper backbone — encoder-only SSL for LFP signals.

Uses OpenAI's Whisper encoder as a feature extractor. Since Whisper is
an encoder-decoder ASR model, we only use the encoder and add a masked
prediction SSL objective on top (L2 regression on masked encoder outputs,
similar to Data2Vec).

Each 3s LFP chunk is padded to 30s by WhisperFeatureExtractor (Whisper
requires exactly 3000 mel frames). The full 1500-frame encoder output
is mean-pooled for downstream embeddings.

Note: Chunk packing was removed — padding approach gives better downstream
accuracy (+6pp) because the padded baseline provides a consistent reference
frame that makes signal perturbations easier for probes to detect.
"""

from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import WhisperConfig, WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from backbones.base import BackboneBase


class WhisperBackbone(BackboneBase):
    """Whisper encoder backbone for SSL pretraining on LFP signals.

    SSL objective: Masked prediction — randomly mask encoder time steps,
    then predict the masked representations using a lightweight projection
    head with L2 loss (similar to Data2Vec).

    Each 3s LFP chunk is padded to 30s by WhisperFeatureExtractor. The full
    1500-frame encoder output is mean-pooled for downstream embeddings.
    """

    ENCODER_FRAMES = 1500       # Whisper encoder output length for 30s

    def __init__(self, config: dict):
        super().__init__()

        self._hidden_dim = config.get("hidden_size", 768)
        model_id = config.get("pretrained_model", "openai/whisper-base")
        rand_init = config.get("rand_init", False)

        # Masking params for SSL
        self.mask_prob = config.get("mask_time_prob", 0.3)
        self.mask_length = config.get("mask_time_length", 10)

        # Whisper encoder + feature extractor
        if rand_init or model_id is None:
            # Create feature extractor with default Whisper params (no download)
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
                dropout=config.get("hidden_dropout", 0.1),
                attention_dropout=config.get("attention_dropout", 0.0),
                activation_dropout=config.get("final_dropout", 0.0),
            )
            self.encoder = WhisperEncoder(whisper_config)
        else:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
            whisper_config = WhisperConfig.from_pretrained(model_id)
            self.encoder = WhisperEncoder(whisper_config)
            # Load only encoder weights from pretrained
            from transformers import WhisperModel
            pretrained = WhisperModel.from_pretrained(model_id)
            self.encoder.load_state_dict(pretrained.encoder.state_dict())
            self._hidden_dim = whisper_config.d_model
            del pretrained

        # Mask embedding (learnable vector replacing masked positions)
        self.mask_emb = nn.Parameter(torch.zeros(self._hidden_dim))
        nn.init.normal_(self.mask_emb, std=0.02)

        # Prediction head for SSL (predicts masked encoder outputs)
        pred_hidden = config.get("predictor_hidden", 256)
        self.predictor = nn.Sequential(
            nn.Linear(self._hidden_dim, pred_hidden),
            nn.GELU(),
            nn.Linear(pred_hidden, self._hidden_dim),
        )

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def needs_resampling(self) -> bool:
        return True  # Expects 16kHz input

    @property
    def needs_normalization(self) -> bool:
        return True  # LFP needs ×3000 global scale for Whisper's log-mel

    def _compute_mel(self, input_values: torch.Tensor) -> torch.Tensor:
        """Convert raw waveform to log-mel spectrogram.

        Each chunk is individually padded to 30s by the feature extractor.

        Args:
            input_values: [B, T] raw waveforms (T ≈ 48000 for 3s @ 16kHz)

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

    def _compute_mask(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute mask indices for SSL training.

        Returns:
            Boolean mask [B, seq_len] where True = masked
        """
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        for b in range(batch_size):
            num_masks = max(1, int(self.mask_prob * seq_len / self.mask_length))
            for _ in range(num_masks):
                start = torch.randint(0, max(1, seq_len - self.mask_length), (1,)).item()
                end = min(start + self.mask_length, seq_len)
                mask[b, start:end] = True

        return mask

    def ssl_forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """Masked prediction SSL objective.

        Each 3s chunk is padded to 30s by the feature extractor. The full
        1500-frame encoder output is used for masked prediction.
        """
        mel = self._compute_mel(input_values)  # [B, 80, 3000]

        # Get target representations (no masking)
        with torch.no_grad():
            target_out = self.encoder(mel).last_hidden_state  # [B, 1500, D]

        batch_size, seq_len, _ = target_out.shape

        # Compute mask
        mask = self._compute_mask(batch_size, seq_len, target_out.device)

        # Re-run encoder (student gets masked input post-hoc)
        encoder_out = self.encoder(mel).last_hidden_state  # [B, 1500, D]

        # Apply mask embedding to masked positions
        masked_out = encoder_out.clone()
        masked_out[mask] = self.mask_emb.to(masked_out.dtype)

        # Predict masked positions
        predictions = self.predictor(masked_out)  # [B, 1500, D]

        # L2 loss on masked positions only
        target = target_out.detach()
        loss = F.mse_loss(predictions[mask], target[mask])

        return loss

    def encode(self, input_values: torch.Tensor) -> torch.Tensor:
        """Extract mean-pooled embeddings.

        Each 3s chunk is padded to 30s by the feature extractor, then the
        full 1500-frame encoder output is mean-pooled.

        Args:
            input_values: [B, T] raw waveform (3s chunks @ 16kHz)

        Returns:
            [B, hidden_dim] per-chunk embeddings
        """
        mel = self._compute_mel(input_values)  # [B, 80, 3000]
        encoder_out = self.encoder(mel).last_hidden_state  # [B, 1500, D]
        return encoder_out.mean(dim=1)  # [B, D]

    def get_optimizer_params(self, lr: float) -> List[Dict[str, Any]]:
        """Separate LR for encoder (pretrained) vs predictor (new)."""
        encoder_lr = lr * 0.1  # Lower LR for pretrained encoder
        return [
            {"params": self.encoder.parameters(), "lr": encoder_lr},
            {"params": self.predictor.parameters(), "lr": lr},
            {"params": [self.mask_emb], "lr": lr},
        ]
