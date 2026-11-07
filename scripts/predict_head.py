#!/usr/bin/env python3
"""Predict from an exported paper linear/MLP decoder and frozen LFP embeddings.

Input NPZ must contain emb, probe_ids and channel_ids. Output coordinates and
standard deviations are AP/DV/ML micrometres. Use trusted checkpoints only.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from lfpformer.paths import activate_reference
from lfpformer.heads import GaussianXYZHead


def predict(
    input_path, checkpoint, output, whitening=None, batch_size=1024, device="cpu"
):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    with np.load(input_path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    x = data.pop("emb").astype(np.float32)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected finite nonempty N x D embeddings")
    for key in ("probe_ids", "channel_ids"):
        if key not in data or len(data[key]) != len(x):
            raise ValueError("Aligned probe_ids and channel_ids are required")
    if whitening is not None:
        with np.load(whitening, allow_pickle=False) as z:
            x = ((x - z["mean"]) @ z["W"].T).astype(np.float16).astype(np.float32)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model" in ck and "kind" in ck:
        activate_reference("benchmark")
        from linear_head_patched import GaussHead

        model = GaussHead(
            ck["kind"],
            d=x.shape[1],
            head=ck.get("head", "gauss"),
            mlp_layers=ck.get("mlp_layers", 1),
        )
        model.load_state_dict(ck["model"], strict=True)
        mean, std = ck["mu_t"], ck["sd_t"]
    elif "head_state_dict" in ck and "y_mean" in ck:
        model = GaussianXYZHead(x.shape[1])
        model.load_state_dict(ck["head_state_dict"], strict=True)
        mean, std = ck["y_mean"], ck["y_std"]
    else:
        raise ValueError(
            "Unsupported head checkpoint. Use the sequence/fusion reference loader for Transformer or manipulator heads."
        )
    mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(std, dtype=torch.float32, device=device)
    if (
        mean.shape != (3,)
        or std.shape != (3,)
        or not torch.isfinite(mean).all()
        or not torch.isfinite(std).all()
        or (std <= 0).any()
    ):
        raise ValueError(
            "Checkpoint target normalization must be finite with positive scale"
        )
    model.to(device).eval()
    mu_out = []
    sigma_out = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            mu, lv = model(torch.from_numpy(x[start : start + batch_size]).to(device))
            mu_out.append((mu * std + mean).cpu().numpy())
            sigma_out.append((torch.exp(0.5 * lv) * std).cpu().numpy())
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, mu=np.concatenate(mu_out), sigma=np.concatenate(sigma_out), **data
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--whitening")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    predict(a.input, a.checkpoint, a.output, a.whitening, a.batch_size, a.device)


if __name__ == "__main__":
    main()
