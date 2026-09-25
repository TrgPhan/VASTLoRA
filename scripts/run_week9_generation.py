"""Run the Week 9 generative matrix using the shared asynchronous LoRA methods."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_week8_classification_matrix as matrix_runner
from week9_artifacts import validate_run


PROFILE_RECIPES = {
    "qwen7b-pilot": "week9_generation_7b_pilot.json",
    "qwen7b-convergence": "week9_generation_7b_convergence.json",
    "qwen7b-rift-diagnostics": "week9_generation_7b_rift_diagnostics.json",
    "qwen7b-seed9102": "week9_generation_7b_seed9102.json",
    "qwen15b-fedlora": "week9_generation_1_5b_fedlora.json",
}


def load_matrix(phase="development", smoke=False, study=None, profile=None):
    matrix = json.loads((ROOT / f"configs/week9_generation_{phase}_matrix.json").read_text())
    if profile is not None:
        if profile not in PROFILE_RECIPES:
            raise ValueError(f"unknown generation profile: {profile}")
        if (phase != "development" and profile != "qwen15b-fedlora") or study is not None:
            raise ValueError(f"{profile} requires development and cannot be combined with a study")
        recipe = json.loads((ROOT / "configs" / PROFILE_RECIPES[profile]).read_text())
        matrix.update(name=recipe["name"], methods=recipe["methods"], seeds=recipe["seeds"])
        matrix["regimes"] = [r for r in matrix["regimes"] if r["name"] in recipe["regimes"]]
        matrix["experiment"].update(recipe.get("experiment_overrides", {}))
        for task in matrix["tasks"]:
            task["model_overrides"].update(recipe["model_overrides"])
            task.update(recipe.get("task_overrides", {}))
        matrix["gates"]["baselines"] = [
            method for method in matrix["gates"]["baselines"] if method in recipe["methods"]
        ]
        if "gates" in recipe:
            matrix["gates"].update(recipe["gates"])
        matrix["notes"].extend(recipe["notes"])
    if study is not None:
        if phase != "development":
            raise ValueError("budget/prompt studies require development, never confirmation")
        recipe = json.loads((ROOT / "configs/week9_generation_budget_study.json").read_text())
        if study not in recipe["prompt_formats"]:
            raise ValueError("unknown generation study")
        matrix.update(name=f"{recipe['name']}-{study}", methods=recipe["methods"], seeds=recipe["seeds"])
        matrix["regimes"] = [r for r in matrix["regimes"] if r["name"] in recipe["regimes"]]
        matrix["experiment"].update(collected_returns=max(recipe["milestones"]),
                                     generation_eval_returns=recipe["milestones"])
        for task in matrix["tasks"]:
            task.update(eval_examples=recipe["eval_examples"],
                        prompt_format=recipe["prompt_formats"][study],
                        eligibility_prompt_formats=list(recipe["prompt_formats"].values()))
        matrix["gates"]["baselines"] = [m for m in matrix["gates"]["baselines"] if m in recipe["methods"]]
        matrix["notes"].extend(recipe["notes"])
    if smoke:
        if phase != "development":
            raise ValueError("smoke uses development data only")
        matrix.update(name="rift-week9-generative-smoke-v4", phase="smoke", seeds=[9001])
        matrix["regimes"] = matrix["regimes"][:1]
        matrix["experiment"].update(warmup_returns=1, collected_returns=5,
                                     calibration_gradient_examples=2, calibration_gate_examples=2,
                                     monitor_examples=2)
        if study or "generation_eval_returns" in matrix["experiment"]:
            matrix["experiment"]["generation_eval_returns"] = [2, 5]
        if study:
            matrix["name"] += f"-{study}-study"
        if profile:
            matrix["name"] = recipe["name"] + "-smoke"
        for task in matrix["tasks"]:
            task.update(max_train_examples=32, eval_examples=4)
    return matrix


def specs(matrix, output):
    for task in matrix["tasks"]:
        base = json.loads((ROOT / task["base_config"]).read_text())
        for regime in matrix["regimes"]:
            config = matrix_runner._build_config(base, task, regime, matrix)
            config["provenance"] = {
                "matrix_name": matrix["name"], "matrix_sha256": matrix_runner._matrix_fingerprint(matrix),
                "phase": matrix["phase"], "task": task["name"], "regime": regime["name"],
            }
            for method in matrix["methods"]:
                directory = output / task["name"] / regime["name"] / method
                job_config = {**config, "output_dir": str(directory)}
                matrix_runner._validate_generated_config(job_config, method)
                for seed in matrix["seeds"]:
                    yield method, seed, job_config, directory / f"{method}_seed{seed}" / "result.json"


class IncompleteRunError(ValueError):
    pass


def checked_result(job, matrix, path=None):
    method, seed, config, result = job
    path = path or result
    if not matrix_runner._completed_result_matches(path, config=config, method=method, seed=seed, matrix=matrix):
        raise ValueError(f"result is not from the matching clean implementation: {path}")
    try:
        validate_run(path, config=config, method=method, seed=seed, matrix=matrix)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise IncompleteRunError(f"incomplete/corrupt artifacts at {path}; --retry-incomplete archives and restarts: {error}") from error


def stop_workers(active):
    for worker in active:
        if worker["process"].poll() is None:
            worker["process"].terminate()
    for worker in active:
        try:
            worker["process"].wait(timeout=15)
        except subprocess.TimeoutExpired:
            worker["process"].kill()
            worker["process"].wait()
        worker["log"].close()


def execute_jobs(jobs, matrix, output, *, gpus, workers_per_gpu=1, max_jobs=None,
                 retry_incomplete=False, resume_roots=(), plan_only=False):
    """Run isolated workers in GPU slots; never change a scientific budget on failure.

    ``workers_per_gpu`` only controls process-level concurrency.  Every worker still
    receives the same frozen config, seed and exclusive model/optimizer state, so it
    is a throughput setting rather than part of the scientific protocol.
    """
    if workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be positive")
    pending, plan = [], []
    for job in jobs:
        method, seed, config, result = job
        importable = False
        if not result.exists():
            for source in resume_roots:
                candidate = source.resolve() / result.relative_to(output)
                if candidate.exists():
                    checked_result(job, matrix, candidate)
                    importable = True
                    if not plan_only:
                        if result.parent.exists():
                            raise RuntimeError(f"partial destination exists; resolve before importing: {result.parent}")
                        shutil.copytree(candidate.parent, result.parent)
                    break
        complete = result.exists()
        if complete:
            try:
                checked_result(job, matrix)
            except IncompleteRunError:
                if not (retry_incomplete or plan_only):
                    raise
                complete = False
        path = output / "job_configs" / f"{config['experiment']['regime_name']}_{method}_{seed}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        command = [sys.executable, "-u", str(matrix_runner.RUNNER), "--config", str(path),
                   "--method", method, "--seed", str(seed)]
        plan.append({"method": method, "seed": seed, "regime": config["experiment"]["regime_name"],
                     "status": "complete" if complete else "importable" if importable else "pending",
                     "command": command, "result": str(result)})
        if not complete and not (plan_only and importable):
            pending.append((job, command))
    (output / "job_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Selected {len(jobs)}; reusable {len(jobs) - len(pending)}; pending {len(pending)}", flush=True)
    if plan_only:
        return
    if max_jobs is not None:
        pending = pending[:max_jobs]
    # Repeating a GPU ID creates independent process slots on that device.  This is
    # intentionally explicit instead of accepting duplicate --gpu flags, which
    # keeps accidental oversubscription out of existing launch commands.
    active = []
    gpu_ids = list(gpus)
    available = [gpu for gpu in gpu_ids for _ in range(workers_per_gpu)]
    print(
        f"Runtime slots: GPUs={gpu_ids}, workers_per_gpu={workers_per_gpu}, "
        f"total_slots={len(available)}",
        flush=True,
    )
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    completed, last_report = 0, time.monotonic()
    try:
        while pending or active:
            while pending and available:
                job, command = pending.pop(0)
                method, seed, config, result = job
                if result.parent.exists():
                    if not retry_incomplete:
                        raise RuntimeError(f"incomplete run exists: {result.parent}; use --retry-incomplete to archive and restart it")
                    archive = output / "incomplete" / f"{result.parent.name}_{time.time_ns()}"
                    result.parent.resolve().relative_to(output.resolve())
                    archive.resolve().relative_to(output.resolve())
                    archive.parent.mkdir(exist_ok=True)
                    result.parent.rename(archive)
                gpu = available.pop(0)
                environment = os.environ.copy()
                if gpu is not None:
                    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                environment.update(TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
                label = f"{config['experiment']['regime_name']}_{method}_{seed}"
                log_path = logs / f"{label}_{time.time_ns()}.log"
                log = log_path.open("w", encoding="utf-8")
                try:
                    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close()
                    raise
                active.append({"job": job, "gpu": gpu, "process": process, "log": log,
                               "log_path": log_path, "start": time.monotonic()})
                print(f"Started {label}, GPU={gpu}, pid={process.pid}; log={log_path}", flush=True)
            for worker in list(active):
                code = worker["process"].poll()
                if code is None:
                    continue
                worker["log"].close()
                if code:
                    raise RuntimeError(f"worker failed ({code}); all workers stopped; inspect {worker['log_path']}")
                # Dirty development can diagnose code, but cannot enter a resumable confirmation cohort.
                method, seed, config, result = worker["job"]
                validate_run(result, config=config, method=method, seed=seed, matrix=matrix)
                (result.parent / "launcher.json").write_text(json.dumps({
                    "wall_seconds": time.monotonic() - worker["start"], "gpu": worker["gpu"],
                    "workers_per_gpu": workers_per_gpu,
                    "log_path": str(worker["log_path"]), "return_code": code,
                }, indent=2), encoding="utf-8")
                active.remove(worker)
                available.append(worker["gpu"])
                completed += 1
                for entry in plan:
                    if entry["result"] == str(result):
                        entry["status"] = "complete"
                (output / "job_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
                print(f"Completed {method} seed {seed}; newly completed={completed}", flush=True)
            if time.monotonic() - last_report >= 60:
                print(f"Progress: completed={completed}, running={len(active)}, queued={len(pending)}", flush=True)
                last_report = time.monotonic()
            if active:
                time.sleep(1)
    finally:
        stop_workers(active)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["development", "confirmation"], default="development")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--study", choices=["budget", "concise"], help="opt-in development budget/prompt study")
    parser.add_argument("--profile", choices=sorted(PROFILE_RECIPES),
                        help="frozen one-seed 7B development profile")
    parser.add_argument("--method", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--regime", action="append")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="write commands and configs without loading a model")
    parser.add_argument("--gpu", action="append", type=int, help="repeat for one independent job per GPU")
    parser.add_argument(
        "--workers-per-gpu", type=int, default=1,
        help="independent worker processes per GPU; changes throughput only, not job configs",
    )
    parser.add_argument("--max-jobs", type=int, help="limit NEW jobs, never the analysis cohort")
    parser.add_argument("--retry-incomplete", action="store_true", help="archive an interrupted run and restart it from its seed")
    parser.add_argument("--resume-root", action="append", type=Path, default=[])
    args = parser.parse_args()
    if sum((args.dry_run, args.prepare_only, args.plan_only)) > 1:
        parser.error("choose only one preflight mode")
    if args.max_jobs is not None and args.max_jobs < 1:
        parser.error("--max-jobs must be positive")
    if args.workers_per_gpu < 1:
        parser.error("--workers-per-gpu must be positive")
    if args.gpu and (len(set(args.gpu)) != len(args.gpu) or min(args.gpu) < 0):
        parser.error("GPU IDs must be distinct and nonnegative")
    args.output_root = args.output_root.resolve()
    matrix = load_matrix(args.phase, args.smoke, args.study, args.profile)
    for selected, declared in ((args.method, matrix["methods"]), (args.seed, matrix["seeds"]),
                               (args.regime, [r["name"] for r in matrix["regimes"]])):
        if selected and set(selected) - set(declared):
            raise ValueError(f"undeclared selection: {set(selected) - set(declared)}")
    jobs = [spec for spec in specs(matrix, args.output_root.resolve())
            if (not args.method or spec[0] in args.method) and (not args.seed or spec[1] in args.seed)
            and (not args.regime or spec[2]["experiment"]["regime_name"] in args.regime)]
    if args.dry_run:
        for method, seed, config, result in jobs:
            shared = matrix_runner._RUNNER_MODULE
            trace = shared.AsyncEventSimulator(
                shared._build_clients(config["experiment"], [[0]] * config["experiment"]["num_clients"]),
                seed=seed, buffer_size=1, schedule_mode="async",
            ).run(max_returns=config["experiment"]["warmup_returns"] + config["experiment"]["collected_returns"])
            late = sum(e.staleness >= config["experiment"]["late_tau"]
                       for e in trace.records[config["experiment"]["warmup_returns"]:])
            print(json.dumps({"method": method, "seed": seed, "regime": config["experiment"]["regime_name"],
                              "measured_late_events": late, "output": str(result)}))
        print(f"Validated {len(jobs)} jobs; phase={matrix['phase']}")
        return
    if (args.phase == "confirmation" and not (args.prepare_only or args.plan_only)
            and matrix_runner._RUNNER_MODULE._git_worktree_dirty() is not False):
        raise RuntimeError("Confirmation requires a clean frozen checkout")
    args.output_root.mkdir(parents=True, exist_ok=True)
    from filelock import FileLock
    with FileLock(str(args.output_root / ".launcher.lock"), timeout=0):
        run_prepared(args, matrix, jobs)


def run_prepared(args, matrix, jobs):
    manifest_path = args.output_root / "matrix.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != matrix:
        raise ValueError("output root contains a different matrix; use a new directory")
    manifest_path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    if args.prepare_only:
        from riftlora.scale.generation import load_instruction_data, reserve_grouped_splits
        data, audit = load_instruction_data(jobs[0][2])
        config = jobs[0][2]
        exp, ds = config["experiment"], config["dataset"]
        required = ds["max_train_examples"] + sum(exp[k] for k in (
            "calibration_gradient_examples", "calibration_gate_examples", "monitor_examples"))
        if len(data["train"]) < required or len(data[ds["eval_split"]]) < ds["eval_examples"]:
            raise ValueError(f"insufficient retained data: {audit}")
        checked, reservations = set(), []
        for _, seed, config, _ in jobs:
            exp, ds = config["experiment"], config["dataset"]
            key = (exp["regime_name"], seed)
            if key in checked:
                continue
            checked.add(key)
            shared = matrix_runner._RUNNER_MODULE
            sampler = shared._reserve_calibration_splits
            options = {}
            if ds.get("reserve_context_groups", False):
                options["reserve_fn"] = sampler
                sampler = reserve_grouped_splits
            train, calibration = sampler(
                data["train"], label_column=ds["label_column"], seed=seed,
                split_sizes=tuple(exp[k] for k in ("calibration_gradient_examples", "calibration_gate_examples", "monitor_examples")),
                max_train_examples=ds["max_train_examples"],
                stratified=exp.get("calibration_sampling") == "stratified", **options,
            )
            partitions = shared._build_partitions(train, label_column=ds["label_column"], experiment=exp, seed=seed)
            reservations.append({"regime": key[0], "seed": seed,
                                 "clients": len(train), "calibration_sizes": [len(d) if d is not None else 0 for d in calibration],
                                 "partition_sizes": [len(p) for p in partitions]})
        audit["reservation_checks"] = reservations
        (args.output_root / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(json.dumps(audit, indent=2))
        return
    execute_jobs(
        jobs, matrix, args.output_root, gpus=args.gpu or [None],
        workers_per_gpu=args.workers_per_gpu, max_jobs=args.max_jobs,
        retry_incomplete=args.retry_incomplete, resume_roots=args.resume_root,
        plan_only=args.plan_only,
    )


if __name__ == "__main__":
    main()
