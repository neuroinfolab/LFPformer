"""Relocate historical defaults; explicit CLI paths always take precedence."""

import os
from pathlib import Path

REFERENCE = Path(__file__).resolve().parent / "_reference"


def legacy_path(path):
    root = (
        Path(os.environ.get("LFPFORMER_WORKSPACE", "workspace")).expanduser().resolve()
    )
    prefixes = [
        "/workspace/ws1/AlphaBrain/Alphabrain_staging/",
        "/workspace/ws2/Alphabrain_staging/",
        "/workspace/ws1/",
        "/workspace/ws2/",
        "/workspace/ws4/",
        "/mnt/work/",
    ]
    for prefix in prefixes:
        if path.startswith(prefix):
            return str(root / path[len(prefix) :])
    return path


def activate_reference(stack="benchmark"):
    import sys

    base = REFERENCE / stack
    for p in [REFERENCE, base, base / "scripts"]:
        sys.path.insert(0, str(p))
    os.environ.setdefault("PB_BENCH", str(base))
    return base
