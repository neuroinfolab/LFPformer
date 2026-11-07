#!/usr/bin/env python3
"""Run a matching encoder and Gaussian per-window, Transformer or fusion decoder.

Input NPZ: signals (N,3750), probe_ids (N,), physical channel_ids (N,),
and geometry_xyz (N,3) in micrometres for Kabsch. No anatomical labels needed.
Use the original training preprocessing and saved whitening transform.
Select --decoder-kind for the checkpoint format. Fusion requires manip_xyz.
"""
import argparse
from pathlib import Path
import numpy as np

from lfpformer.inference import extract_embeddings
from lfpformer.cli import main as postprocess_cli
from predict_head import predict


def run(input_path, config, encoder, decoder, output_dir, whitening=None,
        device="cpu", batch_size=32, method="ibl-kabsch", orientation="ibl",
        decoder_kind="per-window"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    embeddings = out / "embeddings.npz"
    predictions = out / "window_predictions.npz"
    channels = out / "channel_predictions.npz"
    extract_embeddings(input_path, config, encoder, embeddings, batch_size, device,
                       embedding_dtype=np.float32 if decoder_kind == "fusion-linear" else np.float16)
    if decoder_kind == "per-window":
        predict(embeddings, decoder, predictions, whitening, batch_size, device)
    else:
        from predict_sequence import predict as predict_specialized
        predict_specialized(embeddings, decoder, predictions, decoder_kind, whitening, device)
    postprocess_cli(["postprocess", "--input", str(predictions), "--output",
                     str(channels), "--method", method, "--orientation", orientation])
    return channels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--encoder", required=True)
    p.add_argument("--decoder", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--whitening")
    p.add_argument("--decoder-kind", choices=["per-window", "transformer", "fusion-linear", "fusion-transformer"], default="per-window")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--method", choices=["pool", "ibl-kabsch", "allen-line"],
                   default="ibl-kabsch")
    p.add_argument("--orientation", choices=["ibl", "free"], default="ibl")
    a = p.parse_args()
    if a.batch_size < 1:
        p.error("batch-size must be positive")
    result = run(a.input, a.config, a.encoder, a.decoder, a.output_dir,
                 a.whitening, a.device, a.batch_size, a.method, a.orientation, a.decoder_kind)
    print(result)


if __name__ == "__main__":
    main()
