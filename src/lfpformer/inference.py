"""Frozen LFP embeddings using the paper's pinned wav2vec2 implementation."""

from pathlib import Path
import numpy as np
import yaml
from .paths import activate_reference


def extract_embeddings(
    input_path, config_path, checkpoint_path, output_path, batch_size=32, device="cpu",
    embedding_dtype=np.float16,
):
    import torch

    activate_reference("benchmark")
    from backbones import get_backbone
    from utils.preprocessing import preprocess_batch

    config = yaml.safe_load(Path(config_path).read_text())
    config["rand_init"] = True
    model = get_backbone(config.get("backbone", "wav2vec2"), config).to(device)
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ck.get("backbone_state_dict")
    if state is None:
        raise ValueError(
            "Expected backbone_state_dict in trusted checkpoint; use the relevant reference loader for other formats"
        )
    model.load_state_dict(state, strict=True)
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with np.load(input_path, allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    signals = payload.pop("signals")
    if signals.ndim != 2 or signals.shape[1] != 3750:
        raise ValueError(
            "Expected (N,3750) preprocessed 3-second LFP windows at 1250 Hz"
        )
    if not np.isfinite(signals).all() or not len(signals):
        raise ValueError("Nonempty finite windows required")
    embeddings = []
    with torch.inference_mode():
        for i in range(0, len(signals), batch_size):
            x = torch.as_tensor(
                signals[i : i + batch_size], dtype=torch.float32, device=device
            )
            x = preprocess_batch(x, model.needs_resampling, model.needs_normalization)
            embeddings.append(model.encode(x).cpu().numpy().astype(embedding_dtype))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, emb=np.concatenate(embeddings), **payload)
