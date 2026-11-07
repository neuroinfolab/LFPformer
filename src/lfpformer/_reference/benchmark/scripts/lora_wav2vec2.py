#!/usr/bin/env python3
"""Minimal LoRA for HuggingFace Wav2Vec2 attention projections (no peft).

Inject after loading SSL weights into a vanilla backbone. Base Linear weight/bias
are frozen; only lora_A / lora_B train. Forward starts as identity (B = 0).
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """y = W x + b + (alpha/r) * B (A x); W,b frozen."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank

        # Keep adapters on the same device/dtype as the frozen base Linear.
        # (Creating on CPU after backbone.to(cuda) breaks extract encode().)
        device, dtype = base.weight.device, base.weight.dtype
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # A: (r, in), B: (out, r). Zero-B => identity at init.
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, self.in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, self.rank, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        # F.linear(x, A) -> (..., r); F.linear(..., B) -> (..., out)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base + self.scaling * delta


def _wav2vec2_root(backbone: nn.Module):
    if hasattr(backbone, "model") and hasattr(backbone.model, "wav2vec2"):
        return backbone.model.wav2vec2
    if hasattr(backbone, "wav2vec2"):
        return backbone.wav2vec2
    raise AttributeError("Could not find wav2vec2 module on backbone")


def inject_lora(
    backbone: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    targets: Sequence[str] = ("q_proj", "v_proj"),
) -> int:
    """Replace attention Linear modules with LoRALinear; freeze non-LoRA params.

    Returns the number of trainable LoRA parameters.
    """
    if rank < 1:
        return 0
    targets = tuple(targets)
    w2v2 = _wav2vec2_root(backbone)
    encoder = w2v2.encoder
    n_replaced = 0
    for layer in encoder.layers:
        attn = layer.attention
        for name in targets:
            if not hasattr(attn, name):
                raise AttributeError(f"attention missing target {name!r}")
            module = getattr(attn, name)
            if isinstance(module, LoRALinear):
                continue
            if not isinstance(module, nn.Linear):
                raise TypeError(
                    f"expected nn.Linear for {name}, got {type(module)}"
                )
            setattr(attn, name, LoRALinear(module, rank=rank, alpha=alpha))
            n_replaced += 1

    for name, param in backbone.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    n_train = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    if n_replaced == 0:
        raise RuntimeError("inject_lora replaced 0 modules")
    return n_train


def lora_state_dict(backbone: nn.Module) -> dict:
    """Only LoRA adapter tensors (portable; base weights stay in backbone_state_dict)."""
    return {
        k: v.detach().cpu()
        for k, v in backbone.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }


def load_lora_state_dict(backbone: nn.Module, state: dict, strict: bool = True) -> None:
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    lora_missing = [k for k in missing if "lora_A" in k or "lora_B" in k]
    if strict and lora_missing:
        raise RuntimeError(f"Missing LoRA keys: {lora_missing[:8]}")
    if strict and unexpected:
        raise RuntimeError(f"Unexpected LoRA keys: {list(unexpected)[:8]}")


def parse_lora_targets(s: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(s, str):
        parts = [p.strip() for p in s.split(",") if p.strip()]
    else:
        parts = [str(p).strip() for p in s if str(p).strip()]
    if not parts:
        raise ValueError("lora targets empty")
    return tuple(parts)


def apply_lora_from_checkpoint(backbone: nn.Module, ckpt: dict) -> bool:
    """If ckpt has lora_rank>0, inject adapters then load full backbone_state_dict.

    Returns True if LoRA was applied.
    """
    rank = int(ckpt.get("lora_rank", 0) or 0)
    if rank < 1:
        return False
    alpha = float(ckpt.get("lora_alpha", 8.0))
    targets = parse_lora_targets(ckpt.get("lora_targets", "q_proj,v_proj"))
    inject_lora(backbone, rank=rank, alpha=alpha, targets=targets)
    sd = ckpt.get("backbone_state_dict")
    if sd is None:
        raise KeyError("LoRA checkpoint missing backbone_state_dict")
    backbone.load_state_dict(sd, strict=True)
    return True
