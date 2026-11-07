#!/usr/bin/env python3
"""Portable entry point for the published folded direct-XYZ fusion experiment."""
import argparse, sys, json, shutil
from pathlib import Path
from lfpformer.paths import REFERENCE

sys.path.insert(0, str(REFERENCE / "fusion"))
import experiment as E

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("command", choices=["prepare", "train"])
p.add_argument("--out", type=Path, required=True)
p.add_argument("--cache-dir", type=Path, required=True)
p.add_argument("--manip-dir", type=Path, required=True)
p.add_argument("--subject-map", type=Path, required=True)
p.add_argument("--all-subject-map", type=Path, required=True)
p.add_argument(
    "--encoder",
    required=True,
    help="Encoder checkpoint identifier recorded for provenance",
)
p.add_argument("--arm", choices=["lfp_only", "lfp_manip"], default="lfp_manip")
p.add_argument("--seed", type=int, default=1)
a = p.parse_args()
E.ROOT = a.out.resolve()
E.ROOT.mkdir(parents=True, exist_ok=True)
E.SCRIPTS = E.ROOT / "metadata"
E.SCRIPTS.mkdir(exist_ok=True)
shutil.copy2(a.all_subject_map, E.SCRIPTS / ".cache_ibl_session_to_subject.json")
E.CONFIG.update(
    cache_dir=str(a.cache_dir.resolve()),
    manip_dir=str(a.manip_dir.resolve()),
    subject_map=str(a.subject_map.resolve()),
    encoder=a.encoder,
)
if a.command == "prepare":
    E.prepare()
else:
    E.train(a.arm, a.seed)
