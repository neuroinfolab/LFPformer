# Use the Hugging Face models

The [public LFPformer collection](https://huggingface.co/collections/Shubhronyu/lfpformer-large-models-6af10ace16553acb982539dd) contains four Large encoders
from the matched **35,000-step** comparison. They use 24 Transformer layers,
hidden width 1,024, and 16 attention heads. At effective batch size 512, each
checkpoint has seen 17.92 million window presentations, including repeats.
The collection contains fixed-budget checkpoints, not the 100k training endpoints.

| Objective | Model |
|---|---|
| MaskPred | [LFPformer-large-MaskPred-35k](https://huggingface.co/Shubhronyu/LFPformer-large-MaskPred-35k) |
| ChanDistc | [LFPformer-large-ChanDistc-35k](https://huggingface.co/Shubhronyu/LFPformer-large-ChanDistc-35k) |
| ManipDistc | [LFPformer-large-ManipDistc-35k](https://huggingface.co/Shubhronyu/LFPformer-large-ManipDistc-35k) |
| CCFDistc | [LFPformer-large-CCFDistc-35k](https://huggingface.co/Shubhronyu/LFPformer-large-CCFDistc-35k) |

ChanDistc uses within-probe geometry during pretraining; ManipDistc uses logged
insertion geometry; CCFDistc uses histologically registered atlas coordinates.
All four accept LFP alone for embedding extraction. A public download needs no
Hugging Face account, token, or access request.

## Prepare the input and environment

Use Python 3.10 or newer. Install mutually compatible PyTorch and torchaudio
builds for CPU or your CUDA environment (validated: torch 2.9.1 and torchaudio
2.9.1). For a standard pip installation:

```bash
python -m pip install "torch==2.9.1" "torchaudio==2.9.1"
python -m pip install "transformers==4.57.6" "safetensors>=0.7" huggingface_hub numpy
```

Create `windows.npz` with a numeric array named `signals`, shape `(N, 3750)`:
preprocessed single-channel windows of 3 seconds at 1,250 Hz. Preserve the signal
units and dataset-specific preprocessing used during training; see
[data preparation](data.md). The helper does not destripe raw recordings.
Do not apply an audio processor's default per-window waveform normalization.

## Extract embeddings

From the repository root:

```bash
python scripts/extract_hf_embeddings.py \
  --model Shubhronyu/LFPformer-large-ChanDistc-35k \
  --input windows.npz --output outputs/embeddings.npz \
  --device cpu --batch-size 1
```

The first run downloads the model into the Hugging Face cache. Later runs reuse
it. Use `--device cuda` and increase `--batch-size` to fit GPU memory. Set
`HF_HOME` to a writable storage directory if you need to relocate the cache.
Use `--revision <commit-hash>` to pin an exact Hub revision.

The helper resamples each window from 1,250 to 16,000 Hz using torchaudio's sinc
interpolation, averages the final hidden states over time, and saves float32
`emb` of shape `(N, 1024)` in input order. It does not unit-normalize or whiten
the resulting embeddings. Keep your channel/probe metadata aligned separately;
the output NPZ contains only `emb`.

## Load directly with Transformers

This example processes one preprocessed window from your input file:

```python
import numpy as np
import torch
import torchaudio.functional as AF
from transformers import Wav2Vec2ForPreTraining

model_id = "Shubhronyu/LFPformer-large-ChanDistc-35k"
model = Wav2Vec2ForPreTraining.from_pretrained(model_id).eval()
with np.load("windows.npz", allow_pickle=False) as data:
    signals = data["signals"][:1]
assert signals.shape == (1, 3750) and np.isfinite(signals).all()
x = torch.as_tensor(signals, dtype=torch.float32)
with torch.inference_mode():
    x = AF.resample(x, orig_freq=1250, new_freq=16000)
    embeddings = model.wav2vec2(x).last_hidden_state.mean(dim=1)
print(embeddings.shape)  # torch.Size([1, 1024])
```

Loading uses standard Transformers and Safetensors; `trust_remote_code` is not
needed. These models are LFP representation encoders, not speech recognizers.

## Download once and use offline

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Shubhronyu/LFPformer-large-ChanDistc-35k",
    local_dir="checkpoints/LFPformer-large-ChanDistc-35k",
)
```

After the download completes:

```bash
HF_HUB_OFFLINE=1 python scripts/extract_hf_embeddings.py \
  --model checkpoints/LFPformer-large-ChanDistc-35k \
  --input windows.npz --output outputs/embeddings.npz --batch-size 1
```

## Downstream use and compatibility

Train a readout for your task using these embeddings. Fit any whitening or
normalization on the training split and reuse it for validation and test data.
Localization additionally requires a compatible fitted coordinate decoder,
its normalization, and the probe metadata needed for refinement. See
[decoders](decoders.md) and [inference](inference.md).

The Hub folders contain `model.safetensors` and `config.json` for standard
`Wav2Vec2ForPreTraining`. They are not the native `.pt` wrapper format expected
by `lfpformer embed --checkpoint` or `scripts/infer_lfp.py --encoder`.
Use `scripts/extract_hf_embeddings.py` for these releases. The historical
benchmark loader also hardcodes a 256-dimensional codebook projection, whereas
these Large models use 768; converting the file format alone does not resolve
that compatibility difference.

`auxiliary.safetensors` preserves wrapper tensors, including an XYZ head whose
training weight was zero; that head is not a fitted localization decoder.
The supplied restoration helper reconstructs the original tensor dictionary
for compatible research code. There are no trained downstream heads,
whitening artifacts, optimizer states, or raw recordings in the release.

Each Hub repository includes a model card, sanitized training configuration,
preprocessing description, provenance, and checksums. Consult its card for
weight licensing; this code repository's MIT license does not itself grant a
separate license for the released model weights.
