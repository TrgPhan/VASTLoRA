from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vastlora.scale.nll_debug import write_nll_debug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze per-example 3B NLL artifacts")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = write_nll_debug(args.input_dir, args.output_dir)
    print(f"Wrote NLL debug artifacts to {outputs.output_dir}")
    if outputs.missing_runs:
        print("Missing per-example artifacts:")
        for run in outputs.missing_runs:
            print(f"- {run}")
    elif not outputs.paired_summary.empty:
        print(outputs.paired_summary.to_string(index=False))


if __name__ == "__main__":
    main()
