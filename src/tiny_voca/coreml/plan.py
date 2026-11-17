"""Where each op of the Core ML models runs: the share on the Neural Engine, per function.

    python -m tiny_voca.coreml.plan --checkpoint tiny-voca.pt
    python -m tiny_voca.coreml.plan --models coreml [--vocoder]
"""

import argparse
import json
from pathlib import Path
import sys

from .__main__ import UNITS, models_directory


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tiny_voca.coreml.plan", description="Share of the ops on the Neural Engine, per function")
    models = parser.add_mutually_exclusive_group(required=True)
    models.add_argument("--checkpoint", type=Path, help="Inference checkpoint; its cached export (made now if missing)")
    models.add_argument("--models", type=Path, help="Directory written by python -m tiny_voca.coreml.export")
    parser.add_argument("--compute-units", choices=UNITS, help="Plan for these units instead of the exported ones")
    parser.add_argument("--vocoder", action="store_true", help="Also plan the vocoder (it runs on the GPU)")
    args = parser.parse_args(argv)
    if sys.platform != "darwin":
        parser.exit(1, "Core ML runs on macOS only\n")
    import coremltools as ct
    from . import ane
    from .runtime import METADATA
    directory = models_directory(args)
    metadata = json.loads((directory / METADATA).read_text(encoding="utf-8"))
    for kind in ("encoder", "decoder", "vocoder"):
        if kind == "vocoder" and not args.vocoder:
            continue
        entry = metadata["models"][kind]
        units = args.compute_units or entry["compute_units"]
        for name, row in ane.compute_plan_report(directory / entry["package"], getattr(ct.ComputeUnit, units)).items():
            others = ", ".join(f"{op} on {device} x{count}" for (op, device), count in sorted(row["not_ne"].items())) or "-"
            print(f"{kind:8} {name:6} {units:11} ops {row['ops']:5}  on NE {row['ne_share']:7.1%} of ops, "
                  f"{row['ne_cost_share']:7.1%} of estimated cost  elsewhere: {others}")


if __name__ == "__main__":
    main()
