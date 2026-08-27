from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from riftlora.scale.tradeoff import (
    DEFAULT_TRADEOFF_CANDIDATES as DEFAULT_CANDIDATES,
    select_tradeoff,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune the MTIP accuracy/NLL trade-off on a disjoint dev split"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/kaggle_3b_mtip_tradeoff.json"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2025])
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--eval-examples", type=int, default=128)
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument("--eval-shuffle-seed", type=int, default=314159)
    parser.add_argument("--collected-returns", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"tuning output is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = args.output_dir / "logs"
    logs.mkdir()

    jobs = [
        (candidate, seed)
        for seed in args.seeds
        for candidate in DEFAULT_CANDIDATES
    ]
    runner = ROOT / "scripts/run_kaggle_3b.py"
    for wave_start in range(0, len(jobs), len(args.gpus)):
        wave = jobs[wave_start : wave_start + len(args.gpus)]
        running = [
            _launch_job(
                runner,
                args,
                candidate,
                seed,
                gpu=args.gpus[index],
                logs=logs,
            )
            for index, (candidate, seed) in enumerate(wave)
        ]
        while any(process.poll() is None for process, _, _ in running):
            time.sleep(15)
        for (candidate, seed), (process, handle, log_path) in zip(wave, running):
            handle.close()
            if process.returncode != 0:
                tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
                raise RuntimeError(
                    f"{candidate['name']} seed={seed} failed:\n" + "\n".join(tail)
                )

    rows = _load_rows(args.output_dir, args.seeds)
    selection = select_tradeoff(rows)
    _write_csv(args.output_dir / "tradeoff_candidates.csv", rows)
    (args.output_dir / "tradeoff_selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(selection, indent=2, sort_keys=True))


def _launch_job(
    runner: Path,
    args: argparse.Namespace,
    candidate: dict[str, Any],
    seed: int,
    *,
    gpu: int,
    logs: Path,
):
    candidate_output = args.output_dir / str(candidate["name"])
    candidate_output.mkdir(exist_ok=True)
    log_path = logs / f"{candidate['name']}_seed{seed}.log"
    log_handle = log_path.open("w", encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    command = [
        sys.executable,
        str(runner),
        "--config",
        str(args.config),
        "--method",
        str(candidate["method"]),
        "--variant",
        str(candidate["name"]),
        "--seed",
        str(seed),
        "--output-dir",
        str(candidate_output),
        "--eval-examples",
        str(args.eval_examples),
        "--eval-offset",
        str(args.eval_offset),
        "--eval-shuffle-seed",
        str(args.eval_shuffle_seed),
        "--eval-split",
        "train",
        "--reserve-eval-from-train",
        *[str(value) for value in candidate["cli_args"]],
    ]
    if args.collected_returns is not None:
        command.extend(["--collected-returns", str(args.collected_returns)])
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return process, log_handle, log_path


def _load_rows(output_dir: Path, seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in DEFAULT_CANDIDATES:
        for seed in seeds:
            result_path = (
                output_dir
                / str(candidate["name"])
                / f"{candidate['name']}_seed{seed}"
                / "result.json"
            )
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "candidate": candidate["name"],
                    "method": candidate["method"],
                    "seed": seed,
                    "final_accuracy": payload["metrics"]["final_accuracy"],
                    "final_balanced_accuracy": payload["metrics"][
                        "final_balanced_accuracy"
                    ],
                    "final_nll": payload["metrics"]["final_nll"],
                    "final_binary_nll": payload["metrics"]["final_binary_nll"],
                    "final_brier": payload["metrics"]["final_brier"],
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

