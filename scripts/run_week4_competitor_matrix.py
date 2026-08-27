from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "configs/week4_rift_competitor_matrix.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matched RIFT competitor sweeps")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--target", action="append", choices=("sst2", "qnli"))
    parser.add_argument("--method", action="append")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    methods = args.method or matrix["methods"]
    unknown = sorted(set(methods) - set(matrix["methods"]))
    if unknown:
        raise ValueError(f"methods are not declared in the matrix: {unknown}")
    targets = args.target or list(matrix["targets"])

    for target_name in targets:
        target = matrix["targets"][target_name]
        base_path = ROOT / target["base_config"]
        base = json.loads(base_path.read_text(encoding="utf-8"))
        base["experiment"].update(matrix["shared_parameters"])
        for method in methods:
            output_dir = ROOT / f"{target['output_prefix']}_{_output_suffix(method)}"
            validation = output_dir / "week3_diagnostics_validation.json"
            if validation.exists() and not args.force:
                print(f"skip completed {target_name}/{method}: {output_dir}")
                continue
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                prefix=f"riftlora_{target_name}_{method}_",
                encoding="utf-8",
                delete=False,
            ) as handle:
                json.dump(base, handle, indent=2)
                config_path = Path(handle.name)
            command = [
                sys.executable,
                str(ROOT / "scripts/collect_week3_diagnostics.py"),
                "--config",
                str(config_path),
                "--accept-method",
                method,
                "--candidate-suite",
                "core",
                "--output-dir",
                str(output_dir),
                "--device",
                args.device,
            ]
            print(" ".join(command))
            try:
                if not args.dry_run:
                    subprocess.run(command, cwd=ROOT, check=True)
            finally:
                config_path.unlink(missing_ok=True)


def _output_suffix(method: str) -> str:
    return {
        "glora_cache": "glora",
        "fedsteer_cache": "fedsteer",
        "alignfed_calibration": "alignfed",
    }.get(method, method)


if __name__ == "__main__":
    main()

