"""Audit downloaded factor-only artifacts and rebuild reports without training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from run_week8_classification_matrix import _build_config, _matrix_fingerprint
from week8_spectral_suite import ROOT, build_matrix, job_list

RELEASE = "db4ca69024fdd0697f7ed667efbf10f025c558a3"


def fingerprint(config):
    config = dict(config)
    config.pop("output_dir", None)
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def close(actual, expected):
    if not math.isfinite(float(actual)) or not math.isclose(float(actual), float(expected),
                                                           rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(f"metric mismatch: {actual} vs {expected}")


def validate(path, config, job):
    task, regime, method, seed = job
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("task"), payload.get("regime"), payload.get("method"), payload.get("seed")) != job:
        raise ValueError("run identity mismatch")
    if (payload.get("git_commit") != RELEASE or payload.get("git_worktree_dirty") is not False
            or payload.get("factor_implementation") != "persistent_factors_v2"
            or payload.get("schema_version", 0) < 5):
        raise ValueError("release/implementation mismatch")
    if (payload.get("provenance") != config["provenance"]
            or payload.get("config_fingerprint") != fingerprint(config)
            or fingerprint(payload["config"]) != fingerprint(config)):
        raise ValueError("config/provenance mismatch")
    metrics, exp, ds = payload["metrics"], config["experiment"], config["dataset"]
    baseline = None
    for stage in ("baseline", "final"):
        frame = pd.read_csv(path.parent / f"{stage}_eval_details.csv")
        if len(frame) != ds["eval_examples"] or frame.eval_index.tolist() != list(range(len(frame))):
            raise ValueError("incomplete evaluation")
        if not frame["is_correct"].isin([0, 1]).all() or not (frame.class_nll >= 0).all():
            raise ValueError("invalid evaluation values")
        close(metrics[f"{stage}_accuracy"], frame.is_correct.mean())
        close(metrics[f"{stage}_class_nll"], frame.class_nll.mean())
        identity = frame[["eval_index", "text", "true_label"]].to_json(orient="records")
        if baseline is None:
            baseline = identity
        elif baseline != identity:
            raise ValueError("baseline/final evaluation data differs")
    events = pd.read_csv(path.parent / "events.csv")
    total = exp["warmup_returns"] + exp["collected_returns"]
    if len(events) != total or events.event.tolist() != list(range(total)):
        raise ValueError("incomplete event trace")
    if not events.method.eq(method).all():
        raise ValueError("wrong method in event trace")
    if events.measured.tolist() != ([False] * exp["warmup_returns"] + [True] * exp["collected_returns"]):
        raise ValueError("incorrect measurement window")
    measured = events.loc[events.measured]
    for column in ("current_loss", "accepted_loss", "local_loss"):
        if not measured[column].map(math.isfinite).all():
            raise ValueError("nonfinite event losses")
    harmful = measured.accepted_loss > measured.current_loss + exp["harm_epsilon"]
    late = measured.staleness >= exp["late_tau"]
    if not harmful.equals(measured.harmful_update) or not (harmful & late).equals(measured.late_harmful_update):
        raise ValueError("harmful flags do not match losses")
    close(metrics["harmful_update_rate"], harmful.mean())
    close(metrics["late_harmful_update_rate"], harmful[late].mean() if late.any() else 0)
    close(metrics["measured_event_count"], len(measured))
    close(metrics["late_event_count"], late.sum())
    schedule = events[["client_id", "client_rank", "base_version", "arrival_version", "staleness"]].to_json()
    return dict(task=task, regime=regime, method=method, seed=seed,
                accuracy=100 * metrics["final_accuracy"], class_nll=metrics["final_class_nll"],
                harmful=100 * metrics["harmful_update_rate"],
                late_harmful=100 * metrics["late_harmful_update_rate"],
                runtime_minutes=metrics["runtime_seconds"] / 60), (baseline, schedule)


def audit(directory):
    matrix = json.loads((directory / "matrix.json").read_text(encoding="utf-8"))
    if matrix != build_matrix(suite="factor_only"):
        raise ValueError("Downloaded matrix differs from locked factor confirmation")
    tasks = {t["name"]: t for t in matrix["tasks"]}
    regimes = {r["name"]: r for r in matrix["regimes"]}
    rows, pending, identities = [], [], {}
    for job in job_list(matrix):
        task, regime, method, seed = job
        spec = tasks[task]
        cfg = _build_config(json.loads((ROOT / spec["base_config"]).read_text()), spec, regimes[regime], matrix)
        cfg["provenance"] = dict(matrix_name=matrix["name"], matrix_sha256=_matrix_fingerprint(matrix),
                                 task=task, regime=regime, base_config=spec["base_config"])
        path = directory / task / regime / method / f"{method}_seed{seed}" / "result.json"
        try:
            row, identity = validate(path, cfg, job)
        except (OSError, ValueError, KeyError, AttributeError) as error:
            pending.append(dict(job=job, error=str(error)))
            continue
        key = (task, regime, seed)
        if key in identities and identities[key] != identity:
            raise ValueError(f"Methods used different evaluation data or schedules: {key}")
        identities[key] = identity
        rows.append(row)
    return pd.DataFrame(rows), pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    frame, pending = audit(args.directory)
    frame.to_csv(args.directory / "per_seed_results.csv", index=False)
    (args.directory / "pending_jobs_audited.json").write_text(json.dumps(pending, indent=2), encoding="utf-8")
    lines = ["# Week 8: notebook7b4f16575b recovery", "", f"Verified: {len(frame)}/48. Pending/invalid: {len(pending)}.",
             "", f"Source: `{args.directory.as_posix()}`.", f"Training release: `{RELEASE}`.", "",
             "All values below are means over completed seeds; accuracy/harmful are percentages.",
             "Only groups with 6 seeds are complete confirmation groups.", "",
             "| Task | Method | Seeds | Accuracy (%) | Class NLL | Harmful (%) | Late harmful (%) | Runtime (min/job) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    if not frame.empty:
        summary = frame.groupby(["task", "method"], sort=False).agg(
            seeds=("seed", "nunique"), accuracy=("accuracy", "mean"), class_nll=("class_nll", "mean"),
            harmful=("harmful", "mean"), late_harmful=("late_harmful", "mean"), runtime=("runtime_minutes", "mean"))
        summary.to_csv(args.directory / "summary.csv")
        for (task, method), r in summary.iterrows():
            lines.append(f"| {task} | {method} | {int(r.seeds)} | {r.accuracy:.2f} | {r.class_nll:.6f} | {r.harmful:.2f} | {r.late_harmful:.2f} | {r.runtime:.2f} |")
        frame.groupby(["task", "method"]).agg({k: ["mean", "std"] for k in
                                               ("accuracy", "class_nll", "harmful", "late_harmful")}).to_csv(args.directory / "mean_std.csv")
    lines += ["", "## Nguyen nhan loi", "",
              "Kernel log: Finished shard: 48, sau do ModuleNotFoundError: No module named 'riftlora' o cell report.",
              "pip install -e chay trong subprocess; kernel notebook dang mo chua nap editable-package path moi.",
              "Day la loi import/bao cao, khong phai bang chung train fail, OOM hay loi thuat toan.",
              "", "## Xu ly", "",
              "Da them REPO_DIR/src vao sys.path cua kernel va kiem tra package dung checkout.",
              "Giu nguyen training release de tai su dung ket qua cu, khong sua logic method.",
              "Gan output notebook cu vao Add Input, RUN_TRAINING=False de tao lai bang/ZIP.",
              "Queue chi giu job chua hoan thanh sau khi kiem tra config, commit, implementation va du artifacts.",
              "REQUIRE_RESUME=True chan train nham khi chua gan output cu.", "",
              "Audit doi chieu metrics voi 1024 eval rows, 40 returns/32 measured, harmful flags va paired data/trace.",
              "Khong suy ra RIFT thang/thua tu bang nay: can doi chieu cohort RIFT cung protocol, seed va ngan sach."]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:23]))
    if pending:
        print(json.dumps(pending, indent=2))


if __name__ == "__main__":
    main()
