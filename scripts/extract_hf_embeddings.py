"""Extract mean-pooled LFP embeddings using the exported native HF weights."""
import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as AF
from transformers import Wav2Vec2ForPreTraining


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='Local model folder or Hugging Face model ID')
    p.add_argument('--input', required=True, help='NPZ containing signals[N,3750] at 1250 Hz')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--revision', default=None, help='Optional pinned Hugging Face commit')
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('--batch-size must be positive')
    model = Wav2Vec2ForPreTraining.from_pretrained(args.model, revision=args.revision).to(args.device).eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with np.load(args.input, allow_pickle=False) as z:
        signals = z['signals']
    if signals.ndim != 2 or signals.shape[1] != 3750 or not len(signals) or not np.isfinite(signals).all():
        raise ValueError('Expected nonempty finite signals with shape (N,3750)')
    output = []
    with torch.inference_mode():
        for i in range(0, len(signals), args.batch_size):
            x = torch.as_tensor(signals[i:i+args.batch_size], dtype=torch.float32, device=args.device)
            x = AF.resample(x, orig_freq=1250, new_freq=16000)
            emb = model.wav2vec2(x).last_hidden_state.mean(dim=1)
            if not torch.isfinite(emb).all():
                raise ValueError('Nonfinite model embeddings')
            output.append(emb.cpu().numpy())
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, emb=np.concatenate(output))


if __name__ == '__main__':
    main()
