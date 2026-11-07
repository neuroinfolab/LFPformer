#!/usr/bin/env python3
"""Resolve local paths in a supplied paper configuration."""
import argparse
from pathlib import Path
import yaml

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--config", type=Path, required=True)
p.add_argument("--data", type=Path, required=True)
p.add_argument("--validation", type=Path, required=True)
p.add_argument("--results", type=Path, required=True)
p.add_argument("--probe-data", type=Path)
p.add_argument("--max-steps", type=int)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
c = yaml.safe_load(a.config.read_text())
c.update(
    compact_data_path=str(a.data.resolve()),
    val_subset_dir=str(a.validation.resolve()),
    results_dir=str(a.results.resolve()),
)
if a.probe_data:
    c["probe_data_path"] = str(a.probe_data.resolve())
if a.max_steps is not None:
    if a.max_steps <= 0:
        raise ValueError("max-steps must be positive")
    c["max_global_step"] = a.max_steps
if not a.probe_data:
    c.update(probe_every_steps=0, probe_interval=0, embed_interval=0)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(yaml.safe_dump(c, sort_keys=False))
