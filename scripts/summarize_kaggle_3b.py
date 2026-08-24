from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vastlora.scale.reporting import render_verdict_markdown, write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Kaggle 3B scale results")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-variant", default="mtip_adaptive")
    parser.add_argument("--development-status")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_dir / "summary"
    summary, verdict = write_summary(
        args.input_dir,
        output_dir,
        target_variant=args.target_variant,
        development_status=args.development_status,
    )
    print(summary.to_string(index=False))
    print()
    print(render_verdict_markdown(verdict))


if __name__ == "__main__":
    main()
