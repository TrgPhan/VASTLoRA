"""Supplement confirmation v3 with post-hoc Spectral Surgery on the old task setup."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_week8_classification_matrix as matrix_runner

METHOD = "spectral_surgery_posthoc"
SOURCE = ROOT / "configs/rift_core_heldout_confirmation_matrix.json"


def build_matrix():
    original = json.loads(SOURCE.read_text(encoding="utf-8"))
    matrix = copy.deepcopy(original)
    matrix["name"] = "spectral-posthoc-v3-supplement-v1"
    matrix["description"] = "New baseline on the existing confirmation v3 data and seeds."
    matrix["methods"] = [METHOD]
    matrix["experiment"].update(
        spectral_edit_target_modules=["q_proj", "v_proj"],
        spectral_posthoc_base_method="freshness",
    )
    matrix.pop("gates", None)
    matrix["notes"] = [
        "Same task splits, seeds, ranks and budgets as confirmation v3; new implementation commit.",
        "Post-hoc answer-token NLL including EOS; 24 calibration examples; q_proj/v_proj edits.",
        "Adapted Spectral Surgery baseline, not original-paper benchmark reproduction.",
        "Harmful rates describe training returns before editing, not post-hoc filtering safety.",
        "All six seeds are reported. Final evaluation never selects the edit or checkpoint.",
    ]
    return matrix


def build_jobs(matrix, output):
    jobs = []
    for task in matrix["tasks"]:
        base = json.loads((ROOT / task["base_config"]).read_text(encoding="utf-8"))
        for regime in matrix["regimes"]:
            config = matrix_runner._build_config(base, task, regime, matrix)
            config["provenance"] = {
                "matrix_name": matrix["name"],
                "matrix_sha256": matrix_runner._matrix_fingerprint(matrix),
                "task": task["name"], "regime": regime["name"],
                "base_config": task["base_config"], "phase": "confirmation_supplement",
            }
            directory = output / task["name"] / regime["name"] / METHOD
            config["output_dir"] = str(directory)
            matrix_runner._validate_generated_config(config, METHOD)
            for seed in matrix["seeds"]:
                jobs.append({"task": task["name"], "seed": seed, "config": config,
                             "result": directory / f"{METHOD}_seed{seed}" / "result.json"})
    return jobs


def valid_result(job, matrix, path=None):
    import pandas as pd
    path = path or job["result"]
    if not path.exists():
        return False
    try:
        if not matrix_runner._completed_result_matches(
            path, config=job["config"], method=METHOD, seed=job["seed"], matrix=matrix,
        ):
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        if not (0 <= metrics["final_accuracy"] <= 1 and
                math.isfinite(metrics["final_class_nll"]) and metrics["final_class_nll"] >= 0):
            return False
        if payload["spectral_posthoc"]["edit_objective"] != "answer_token_nll_including_eos":
            return False
        for name in ("baseline_eval_details", "final_eval_details"):
            if len(pd.read_csv(path.parent / f"{name}.csv")) != job["config"]["dataset"]["eval_examples"]:
                return False
        exp = job["config"]["experiment"]
        return len(pd.read_csv(path.parent / "events.csv")) == exp["warmup_returns"] + exp["collected_returns"]
    except (ValueError, TypeError, KeyError, OSError):
        return False


def summarize(jobs, matrix, output):
    import pandas as pd
    rows = []
    for job in jobs:
        if not valid_result(job, matrix):
            continue
        payload = json.loads(job["result"].read_text(encoding="utf-8"))
        m, pre = payload["metrics"], payload["spectral_posthoc"]["pre_edit_metrics"]
        rows.append({
            "task": job["task"], "method": METHOD, "seed": job["seed"],
            "accuracy_pct": 100 * m["final_accuracy"], "class_nll": m["final_class_nll"],
            "pre_edit_accuracy_pct": 100 * pre["accuracy"], "pre_edit_class_nll": pre["class_nll"],
            "edit_accuracy_delta_pp": 100 * (m["final_accuracy"] - pre["accuracy"]),
            "edit_class_nll_delta": m["final_class_nll"] - pre["class_nll"],
            "harmful_pct": 100 * m["harmful_update_rate"],
            "late_harmful_pct": (100 * m["late_harmful_update_rate"]
                                 if m["late_event_count"] else None),
            "late_event_count": m["late_event_count"], "runtime_min": m["runtime_seconds"] / 60,
            "git_commit": payload["git_commit"],
        })
    missing = [{"task": j["task"], "seed": j["seed"]} for j in jobs
               if not valid_result(j, matrix)]
    (output / "completion.json").write_text(json.dumps({
        "expected": len(jobs), "completed": len(rows), "missing": missing,
    }, indent=2), encoding="utf-8")
    if not rows:
        print(f"Completed 0/{len(jobs)}", flush=True)
        return
    runs = pd.DataFrame(rows).sort_values(["task", "seed"])
    runs.to_csv(output / "runs.csv", index=False)
    summary = runs.groupby("task").agg(
        seeds=("seed", "count"), accuracy_mean_pct=("accuracy_pct", "mean"),
        accuracy_sd_pct=("accuracy_pct", "std"), class_nll_mean=("class_nll", "mean"),
        class_nll_sd=("class_nll", "std"), harmful_mean_pct=("harmful_pct", "mean"),
        late_harmful_mean_pct=("late_harmful_pct", "mean"),
        edit_accuracy_delta_pp=("edit_accuracy_delta_pp", "mean"),
        edit_class_nll_delta=("edit_class_nll_delta", "mean"),
    ).reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    report = ["# Spectral post-hoc: confirmation v3 supplement", "",
              f"Completed {len(rows)}/{len(jobs)}. All declared seeds remain in the manifest.", "",
              "Harmful/late harmful concern training before the single post-hoc edit.", "",
              "Same v3 task setup; this is an adapted baseline, not original-paper reproduction.", ""]
    for title, fields in (
        ("Accuracy", ["seeds", "accuracy_mean_pct", "accuracy_sd_pct", "edit_accuracy_delta_pp"]),
        ("Class NLL", ["seeds", "class_nll_mean", "class_nll_sd", "edit_class_nll_delta"]),
        ("Harmful (pre-edit training)", ["seeds", "harmful_mean_pct", "late_harmful_mean_pct"]),
    ):
        report.extend([f"## {title}", "", summary[["task", *fields]].to_markdown(index=False), ""])
    (output / "results.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Completed {len(rows)}/{len(jobs)}\n{summary.to_string(index=False)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", action="append", type=int)
    parser.add_argument("--resume-root", action="append", type=Path, default=[])
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        execute(args)
        return
    from filelock import FileLock
    args.output_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output_root / ".launcher.lock"), timeout=0):
        execute(args)


def execute(args):
    matrix, output = build_matrix(), args.output_root.resolve()
    jobs = build_jobs(matrix, output)
    if args.dry_run:
        print(json.dumps({"jobs": len(jobs), "method": METHOD, "seeds": matrix["seeds"],
                          "tasks": [t["name"] for t in matrix["tasks"]],
                          "example_config": jobs[0]["config"]}, indent=2))
        return
    if matrix_runner._RUNNER_MODULE._git_worktree_dirty() is not False:
        raise RuntimeError("Use a clean pinned checkout for this cohort.")
    output.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(jobs, matrix, output)
        return
    gpu_ids = args.gpu or [0]
    import torch
    if len(set(gpu_ids)) != len(gpu_ids) or any(g < 0 or g >= torch.cuda.device_count() for g in gpu_ids):
        raise ValueError("GPU IDs must be unique and available.")
    if args.max_jobs is not None and args.max_jobs < 1:
        raise ValueError("max-jobs must be positive")
    pending = []
    for job in jobs:
        path = job["result"]
        if path.exists() and not valid_result(job, matrix):
            raise RuntimeError(f"Existing result is incompatible or incomplete; use a separate output root: {path}")
        if not path.exists():
            for source in args.resume_root:
                candidate = source / path.relative_to(output)
                if valid_result(job, matrix, candidate):
                    if path.parent.exists():
                        raise RuntimeError(f"Partial local run directory blocks import: {path.parent}")
                    shutil.copytree(candidate.parent, path.parent)
                    print(f"Imported {job['task']}/{job['seed']}", flush=True)
                    break
        if valid_result(job, matrix):
            print(f"Skip {job['task']}/{job['seed']}", flush=True)
        else:
            pending.append(job)
    pending = pending[:args.max_jobs] if args.max_jobs else pending
    manifest = output / "manifest"
    manifest.mkdir(exist_ok=True)
    (manifest / "matrix.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    (manifest / "pip-freeze.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    active = {}
    try:
        last_report = 0
        while pending or active:
            for gpu, (process, handle, job, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                del active[gpu]
                if code != 0 or not valid_result(job, matrix):
                    print(log.read_text(encoding="utf-8", errors="replace")[-12000:], flush=True)
                    raise RuntimeError(f"Worker failed: {job['task']}/{job['seed']}, code={code}; log={log}")
                print(f"Finished {job['task']}/{job['seed']} GPU {gpu}", flush=True)
                summarize(jobs, matrix, output)
            for gpu in gpu_ids:
                if gpu in active or not pending:
                    continue
                job = pending.pop(0)
                stem = f"{job['task']}_{job['seed']}"
                config_path = manifest / f"{stem}.json"
                config_path.write_text(json.dumps(job["config"], indent=2), encoding="utf-8")
                log = manifest / f"{stem}_{time.time_ns()}.log"
                handle = log.open("w", encoding="utf-8")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false")
                command = [sys.executable, "-u", str(matrix_runner.RUNNER), "--config", str(config_path),
                           "--method", METHOD, "--seed", str(job["seed"])]
                try:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                except BaseException:
                    handle.close()
                    raise
                active[gpu] = process, handle, job, log
                print(f"Started {stem} GPU {gpu}; log={log}", flush=True)
            if time.monotonic() - last_report >= 60:
                print(f"Active: {[v[2]['task'] + '/' + str(v[2]['seed']) for v in active.values()]}; queued={len(pending)}", flush=True)
                last_report = time.monotonic()
            if active:
                time.sleep(2)
    finally:
        for process, _, _, _ in active.values():
            if process.poll() is None:
                process.terminate()
        for process, handle, _, _ in active.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            handle.close()
        summarize(jobs, matrix, output)


if __name__ == "__main__":
    main()
