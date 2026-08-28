from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "configs/week8_rift_classification_matrix.json"
RUNNER = ROOT / "scripts/run_kaggle_3b.py"
_RUNNER_MODULE = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reproducible Week 8 GLUE 3B classification matrix"
    )
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--task", action="append")
    parser.add_argument("--regime", action="append")
    parser.add_argument("--method", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    methods = args.method or list(matrix["methods"])
    declared_methods = set(matrix["methods"])
    unknown_methods = sorted(set(methods) - declared_methods)
    if unknown_methods:
        raise ValueError(f"methods are not declared in the matrix: {unknown_methods}")

    tasks = matrix["tasks"]
    selected_tasks = args.task or [task["name"] for task in tasks]
    tasks_by_name = {task["name"]: task for task in tasks}
    unknown_tasks = sorted(set(selected_tasks) - set(tasks_by_name))
    if unknown_tasks:
        raise ValueError(f"tasks are not declared in the matrix: {unknown_tasks}")

    regimes = matrix["regimes"]
    selected_regimes = args.regime or [regime["name"] for regime in regimes]
    regimes_by_name = {regime["name"]: regime for regime in regimes}
    unknown_regimes = sorted(set(selected_regimes) - set(regimes_by_name))
    if unknown_regimes:
        raise ValueError(f"regimes are not declared in the matrix: {unknown_regimes}")

    seeds = args.seed or [int(seed) for seed in matrix["seeds"]]
    validated_specs: set[tuple[str, str, str]] = set()
    for task_name in selected_tasks:
        task = tasks_by_name[task_name]
        base_path = ROOT / task["base_config"]
        base = json.loads(base_path.read_text(encoding="utf-8"))
        for regime_name in selected_regimes:
            regime = regimes_by_name[regime_name]
            for method in methods:
                for seed in seeds:
                    config = _build_config(base, task, regime, matrix)
                    spec_key = (task_name, regime_name, method)
                    if spec_key not in validated_specs:
                        _validate_generated_config(config, method)
                        validated_specs.add(spec_key)
                    output_dir = (
                        ROOT
                        / "outputs/week8_classification_matrix"
                        / str(task.get("run_name", task_name))
                        / regime_name
                        / method
                    )
                    result_path = output_dir / f"{method}_seed{seed}" / "result.json"
                    if result_path.exists() and not args.force:
                        print(f"skip completed {task_name}/{regime_name}/{method}/seed{seed}")
                        continue

                    config["output_dir"] = str(output_dir)
                    command = [
                        sys.executable,
                        str(RUNNER),
                        "--config",
                        "{CONFIG}",
                        "--method",
                        method,
                        "--seed",
                        str(seed),
                    ]
                    print(
                        " ".join(command).replace(
                            "{CONFIG}", f"<generated {task_name}/{regime_name}>"
                        )
                    )
                    if args.dry_run:
                        continue
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        suffix=".json",
                        prefix=f"riftlora_week8_{task_name}_{regime_name}_",
                        encoding="utf-8",
                        delete=False,
                    ) as handle:
                        json.dump(config, handle, indent=2)
                        config_path = Path(handle.name)
                    command[3] = str(config_path)
                    try:
                        subprocess.run(command, cwd=ROOT, check=True)
                    finally:
                        config_path.unlink(missing_ok=True)


def _build_config(
    base: dict[str, Any],
    task: dict[str, Any],
    regime: dict[str, Any],
    matrix: dict[str, Any],
) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    dataset = config["dataset"]
    dataset.update(
        {
            key: value
            for key, value in task.items()
            if key not in {"name", "base_config", "output_prefix"}
        }
    )
    experiment = config["experiment"]
    experiment.update(
        {
            "num_clients": len(regime["client_ranks"]),
            "client_ranks": list(regime["client_ranks"]),
            "compute_times": list(regime["compute_times"]),
            "partition_mode": regime["partition_mode"],
            "regime_name": regime["name"],
            "buffer_size": int(matrix["runner"].get("buffer_size", 1)),
            "schedule_mode": str(matrix["runner"].get("schedule_mode", "async")),
        }
    )
    return config


def _validate_generated_config(config: dict[str, Any], method: str) -> None:
    global _RUNNER_MODULE
    if _RUNNER_MODULE is None:
        spec = importlib.util.spec_from_file_location("riftlora_week8_runner", RUNNER)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load runner from {RUNNER}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _RUNNER_MODULE = module
    _RUNNER_MODULE._validate_config(config, method)


if __name__ == "__main__":
    main()
