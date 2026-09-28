"""Freeze and execute opt-in causal controls without overwriting old experiments."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from riftlora.scale.causal_study import fingerprint


def source_manifest(root=ROOT):
    # Normalize line endings for Windows -> Linux checkout portability.
    return {
        p.relative_to(root).as_posix(): fingerprint(p.read_text(encoding="utf-8"))
        for directory in ("src", "scripts")
        for p in sorted((root / directory).rglob("*.py"))
    }


def merge(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def build_jobs(spec, *, suite, phase, tasks=None, variants=None):
    from run_week8_classification_matrix import (
        _build_config,
        _validate_generated_config,
    )

    matrix = json.loads((ROOT / spec["base_matrix"]).read_text(encoding="utf-8"))
    selection = spec["suites"][suite]
    if phase == "confirmation" and suite == "tuning":
        raise ValueError("tuning cannot use confirmation")
    task_names = tasks or [t["name"] for t in matrix["tasks"]]
    variant_names = variants or selection["variants"]
    if not set(task_names) <= {t["name"] for t in matrix["tasks"]}:
        raise ValueError("unknown task")
    if not set(variant_names) <= set(selection["variants"]):
        raise ValueError("variant not declared in suite")
    if len(set(task_names)) != len(task_names) or len(set(variant_names)) != len(
        variant_names
    ):
        raise ValueError("duplicate selectors")
    phase_spec = spec["phases"][phase]
    seeds = phase_spec["seeds"]
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonempty nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seeds")
    if type(phase_spec["eval_examples"]) is not int or phase_spec["eval_examples"] < 1:
        raise ValueError("eval_examples must be a positive integer")
    if not selection["regimes"] or len(set(selection["regimes"])) != len(
        selection["regimes"]
    ):
        raise ValueError("regimes must be nonempty and unique")
    jobs = []
    for task_name in task_names:
        task = next(t for t in matrix["tasks"] if t["name"] == task_name)
        for regime_name in selection["regimes"]:
            regime = {
                "name": regime_name,
                "client_ranks": [2, 4, 8, 4],
                **spec["regimes"][regime_name],
            }
            base = json.loads((ROOT / task["base_config"]).read_text(encoding="utf-8"))
            common = _build_config(base, task, regime, matrix)
            merge(common["experiment"], spec["experiment"])
            common["experiment"]["study_fixed_gate_budget"] = True
            common["experiment"]["methods"] = sorted(
                {spec["variants"][v]["method"] for v in variant_names}
            )
            common["experiment"]["seeds"] = phase_spec["seeds"]
            data = common["dataset"]
            data["study_content_policy"] = spec["content_policy"]
            data["study_eval_windows"] = [
                [spec["eval_offsets"][task_name][p], spec["phases"][p]["eval_examples"]]
                for p in ("development", "confirmation")
            ]
            data["client_objective"] = selection.get("client_objective", "label_nll")
            data["eval_examples"] = phase_spec["eval_examples"]
            window_phase = "development" if phase == "smoke" else phase
            data["eval_offset"] = spec["eval_offsets"][task_name][window_phase]
            if task_name == "sst2":
                data["study_reserved_train_windows"] = [
                    [
                        spec["eval_offsets"][task_name][p],
                        spec["phases"][p]["eval_examples"],
                    ]
                    for p in ("development", "confirmation")
                ]
            if phase == "smoke":
                common["experiment"].update(
                    warmup_returns=1,
                    collected_returns=2,
                    calibration_gradient_examples=2,
                    calibration_gate_examples=2,
                    monitor_examples=2,
                )
                data["max_train_examples"] = 16
            for variant_name in variant_names:
                variant = spec["variants"][variant_name]
                config = copy.deepcopy(common)
                merge(config["experiment"], variant["experiment"])
                config["causal_study"] = {
                    "name": spec["name"],
                    "phase": phase,
                    "suite": suite,
                    "variant": variant_name,
                    "heldout_history": "not_certified_by_code",
                }
                config["provenance"] = {
                    "task": task_name,
                    "regime": regime_name,
                    "phase": phase,
                    "study_sha256": fingerprint(spec),
                }
                _validate_generated_config(config, variant["method"])
                for seed in phase_spec["seeds"]:
                    if type(seed) is not int:
                        raise ValueError("seed must be integer")
                    jobs.append(
                        {
                            "task": task_name,
                            "regime": regime_name,
                            "variant": variant_name,
                            "method": variant["method"],
                            "seed": seed,
                            "config": copy.deepcopy(config),
                        }
                    )
    return jobs


def freeze(spec, *, suite, phase, tasks=None, variants=None, attestation=None):
    if phase == "confirmation" and (not attestation or not attestation.strip()):
        raise ValueError(
            "confirmation requires a held-out audit note; changing seeds alone is insufficient"
        )
    sources = source_manifest()
    jobs = build_jobs(spec, suite=suite, phase=phase, tasks=tasks, variants=variants)
    for job in jobs:
        job["config"]["causal_study"].update(
            source_sha256=fingerprint(sources), heldout_audit_note=attestation
        )
        if phase == "confirmation":
            for section in ("model", "dataset"):
                if not job["config"][section].get("revision"):
                    raise ValueError(
                        "confirmation requires pinned model and dataset revisions"
                    )
    payload = {
        "schema": 1,
        "phase": phase,
        "suite": suite,
        "sources": sources,
        "jobs": jobs,
    }
    payload["manifest_sha256"] = fingerprint(payload)
    return payload


def validate_manifest(plan):
    payload = {k: v for k, v in plan.items() if k != "manifest_sha256"}
    if plan.get("schema") != 1 or plan.get("manifest_sha256") != fingerprint(payload):
        raise ValueError("invalid/edited frozen manifest")
    if plan["sources"] != source_manifest():
        raise ValueError(
            "source changed after freeze; use a new reviewed manifest/output root"
        )


def completed_matches(path, job, config):
    from run_kaggle_3b import _config_fingerprint

    r = json.loads(path.read_text(encoding="utf-8"))
    metrics = r.get("metrics", {})
    audit = r.get("causal_study", {})
    for key in (
        "final_accuracy",
        "final_class_nll",
        "harmful_update_rate",
        "acceptance_rate",
    ):
        if type(metrics.get(key)) not in (int, float) or not math.isfinite(
            metrics[key]
        ):
            return False
    if not all(
        audit.get(key)
        for key in (
            "data_sha256",
            "schedule_sha256",
            "partitions_sha256",
            "environment",
        )
    ):
        return False
    return (
        r.get("method") == job["method"]
        and r.get("variant") == job["variant"]
        and r.get("seed") == job["seed"]
        and r.get("config_fingerprint") == _config_fingerprint(config)
        and r.get("causal_study", {}).get("protocol") == config["causal_study"]
        and r.get("metrics", {}).get("measured_event_count")
        == config["experiment"]["collected_returns"]
    )


def execute(plan, output_root, *, dry_run=False, shard_index=0, shard_count=1):
    validate_manifest(plan)
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard")
    for index, job in enumerate(plan["jobs"]):
        if index % shard_count != shard_index:
            continue
        parent = (
            output_root
            / plan["phase"]
            / plan["suite"]
            / job["task"]
            / job["regime"]
            / job["variant"]
        )
        directory = parent / f"{job['variant']}_seed{job['seed']}"
        path = directory / "result.json"
        config = copy.deepcopy(job["config"])
        config["output_dir"] = str(parent.resolve())
        if path.exists():
            if not completed_matches(path, job, config):
                raise ValueError(
                    f"existing result does not match; refusing overwrite: {path}"
                )
            print(f"SKIP {directory}", flush=True)
            continue
        print(f"{'PENDING' if dry_run else 'RUN'} {directory}", flush=True)
        if dry_run:
            continue
        validate_manifest(plan)
        directory.mkdir(parents=True, exist_ok=True)
        lock = directory / "running.lock"
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(fd, str(os.getpid()).encode())
            cfg = directory / "resolved_config.json"
            if cfg.exists() and json.loads(cfg.read_text(encoding="utf-8")) != config:
                raise ValueError(f"partial output config differs: {cfg}")
            cfg.write_text(json.dumps(config, indent=2), encoding="utf-8")
            command = [
                sys.executable,
                str(ROOT / "scripts/run_kaggle_3b.py"),
                "--config",
                str(cfg.resolve()),
                "--method",
                job["method"],
                "--variant",
                job["variant"],
                "--seed",
                str(job["seed"]),
            ]
            with (directory / "run.log").open("a", encoding="utf-8") as log:
                subprocess.run(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            validate_manifest(plan)
            if not path.exists() or not completed_matches(path, job, config):
                raise RuntimeError(f"incomplete result; see {directory / 'run.log'}")
        finally:
            os.close(fd)
            lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", type=Path, default=ROOT / "configs/rift_core_causal_study.json"
    )
    parser.add_argument(
        "--suite",
        choices=("attribution", "delay", "objective", "tuning"),
        default="attribution",
    )
    parser.add_argument(
        "--phase",
        choices=("smoke", "development", "confirmation"),
        default="development",
    )
    parser.add_argument("--task", action="append")
    parser.add_argument("--variant", action="append")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--freeze", type=Path)
    action.add_argument("--execute", type=Path)
    parser.add_argument("--heldout-audit-note")
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs/rift_core_causal_v2"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.execute:
        if args.task or args.variant:
            parser.error("execute uses frozen jobs; selectors apply only before freeze")
        execute(
            json.loads(args.execute.read_text(encoding="utf-8")),
            args.output_root,
            dry_run=args.dry_run,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
        return
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    plan = freeze(
        spec,
        suite=args.suite,
        phase=args.phase,
        tasks=args.task,
        variants=args.variant,
        attestation=args.heldout_audit_note,
    )
    print(
        json.dumps(
            {
                "jobs": len(plan["jobs"]),
                "phase": args.phase,
                "suite": args.suite,
                "variants": sorted({j["variant"] for j in plan["jobs"]}),
                "manifest_sha256": plan["manifest_sha256"],
            },
            indent=2,
        )
    )
    if args.freeze:
        args.freeze.parent.mkdir(parents=True, exist_ok=True)
        archive = args.freeze.with_suffix(".sources.zip")
        if args.freeze.exists() or archive.exists():
            raise FileExistsError(
                "freeze artifacts already exist; choose a new manifest path"
            )
        # Preserve exact normalized source, including uncommitted additions.
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
            for name, expected in plan["sources"].items():
                text = (ROOT / name).read_text(encoding="utf-8")
                if fingerprint(text) != expected:
                    raise ValueError("source changed while creating release archive")
                z.writestr(name, text)
        with args.freeze.open("x", encoding="utf-8") as f:
            json.dump(plan, f, indent=2)
    # No --execute means no training, even when --dry-run was omitted.


if __name__ == "__main__":
    main()
