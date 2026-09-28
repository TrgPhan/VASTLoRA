"""Paired descriptive analysis; no automatic paper GO or seed selection."""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
import statistics

from run_rift_causal_study import completed_matches
from riftlora.scale.causal_study import fingerprint


def summarize_pairs(values):
    """Pointwise paired t interval conditional on the evaluated split (n is small)."""
    from scipy.stats import t

    n = len(values)
    mean = statistics.mean(values) if n else None
    ci = None
    if n >= 2:
        width = float(t.ppf(0.975, n - 1)) * statistics.stdev(values) / math.sqrt(n)
        ci = [mean - width, mean + width]
    return {"n": n, "mean_delta": mean, "pointwise_95pct_t_interval": ci}


def analyze(plan, root):
    if plan.get("manifest_sha256") != fingerprint(
        {k: v for k, v in plan.items() if k != "manifest_sha256"}
    ):
        raise ValueError("manifest digest mismatch")
    rows, missing = [], []
    for job in plan["jobs"]:
        identity = (job["task"], job["regime"], job["variant"], job["seed"])
        parent = (
            root
            / plan["phase"]
            / plan["suite"]
            / job["task"]
            / job["regime"]
            / job["variant"]
        )
        path = parent / f"{job['variant']}_seed{job['seed']}" / "result.json"
        if not path.exists():
            missing.append(identity)
            continue
        config = {**job["config"], "output_dir": str(parent.resolve())}
        if not completed_matches(path, job, config):
            raise ValueError(f"stale/mismatched result: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        metrics = result["metrics"]
        for key in (
            "final_accuracy",
            "final_class_nll",
            "harmful_update_rate",
            "acceptance_rate",
        ):
            if not isinstance(metrics.get(key), (int, float)) or not math.isfinite(
                metrics[key]
            ):
                raise ValueError(f"nonfinite/missing {key}: {path}")
        rows.append(
            {
                "task": job["task"],
                "regime": job["regime"],
                "variant": job["variant"],
                "seed": job["seed"],
                "metrics": metrics,
                "audit": result["causal_study"],
            }
        )
    groups = collections.defaultdict(list)
    pair_groups = collections.defaultdict(dict)
    for row in rows:
        groups[row["task"], row["regime"], row["variant"]].append(row)
        bucket = pair_groups[row["task"], row["regime"], row["seed"]]
        if row["variant"] in bucket:
            raise ValueError("duplicate result identity")
        bucket[row["variant"]] = row
    for bucket in pair_groups.values():
        first = next(iter(bucket.values()))["audit"]
        for row in bucket.values():
            for key in ("data_sha256", "schedule_sha256", "partitions_sha256"):
                if row["audit"].get(key) != first.get(key) or not first.get(key):
                    raise ValueError(f"unpaired data/schedule/partition: {key}")
            if row["audit"]["environment"] != first["environment"]:
                raise ValueError(
                    "unpaired dependency environments; reconcile versions before pooling"
                )
    summaries = []
    metric_keys = (
        "final_accuracy",
        "final_class_nll",
        "harmful_update_rate",
        "acceptance_rate",
        "runtime_seconds",
        "server_fit_seconds",
        "server_fit_examples_seen",
        "gate_loss_evaluations",
        "server_fit_backward_calls",
        "max_server_fit_parameters",
    )
    for (task, regime, variant), group in sorted(groups.items()):
        summary = {
            "task": task,
            "regime": regime,
            "variant": variant,
            "n": len(group),
            "metrics": {},
        }
        for key in metric_keys:
            values = [r["metrics"][key] for r in group if key in r["metrics"]]
            summary["metrics"][key] = {
                "mean": statistics.mean(values) if values else None,
                "sd": statistics.stdev(values) if len(values) > 1 else None,
            }
        late_counts = [r["metrics"]["late_event_count"] for r in group]
        summary["late_event_counts"] = late_counts
        summary["late_harmful_mean"] = (
            statistics.mean(r["metrics"]["late_harmful_update_rate"] for r in group)
            if all(late_counts)
            else None
        )
        summaries.append(summary)
    pairs = []
    for task, regime, variant in sorted(groups):
        if variant == "core":
            continue
        expected = {
            j["seed"]
            for j in plan["jobs"]
            if j["task"] == task
            and j["regime"] == regime
            and j["variant"] in {"core", variant}
        }
        paired = []
        for seed in sorted(expected):
            bucket = pair_groups.get((task, regime, seed), {})
            if "core" in bucket and variant in bucket:
                paired.append((seed, bucket["core"], bucket[variant]))
        comparison = {
            "task": task,
            "regime": regime,
            "comparator": variant,
            "complete": len(paired) == len(expected),
            "seeds": [r[0] for r in paired],
            "metrics": {},
        }
        for key in ("final_accuracy", "final_class_nll", "harmful_update_rate"):
            values = [a["metrics"][key] - b["metrics"][key] for _, a, b in paired]
            comparison["metrics"][key] = summarize_pairs(values)
        pairs.append(comparison)
    return {
        "manifest_sha256": plan["manifest_sha256"],
        "phase": plan["phase"],
        "missing": missing,
        "completed": len(rows),
        "expected": len(plan["jobs"]),
        "summaries": summaries,
        "pairs": pairs,
        "interpretation": "Core minus comparator. Pointwise paired t intervals, no multiplicity adjustment; small n. Conditional on this split, not a population/independent-test certificate. No pooling MNLI views or seed cherry-picking.",
    }


def markdown(report):
    lines = [
        "# RIFT causal study results",
        "",
        f"Completed: {report['completed']}/{report['expected']}. Phase: {report['phase']}.",
        "",
        "Partial groups are descriptive only. No automatic GO verdict.",
        "",
    ]
    for title, key, scale in (
        ("Accuracy", "final_accuracy", 100),
        ("Class NLL", "final_class_nll", 1),
        ("Harmful updates", "harmful_update_rate", 100),
    ):
        lines += [
            f"## {title}",
            "",
            "| Task | Regime | Variant | n | Mean | SD |",
            "|---|---|---|---:|---:|---:|",
        ]
        for s in report["summaries"]:
            value = s["metrics"][key]
            sd = f"{value['sd'] * scale:.6f}" if value["sd"] is not None else "N/A"
            lines.append(
                f"| {s['task']} | {s['regime']} | {s['variant']} | {s['n']} | {value['mean'] * scale:.6f} | {sd} |"
            )
        lines.append("")
    lines += [
        "## Paired differences",
        "",
        report["interpretation"],
        "",
        "| Task | Regime | Comparator | n | Complete | NLL delta | Pointwise 95% t interval |",
        "|---|---|---|---:|---|---:|---|",
    ]
    for p in report["pairs"]:
        m = p["metrics"]["final_class_nll"]
        lines.append(
            f"| {p['task']} | {p['regime']} | {p['comparator']} | {m['n']} | {p['complete']} | {m['mean_delta']} | {m['pointwise_95pct_t_interval']} |"
        )
    lines += [
        "",
        "Budget, acceptance, late-event denominators and missing job identities are retained in the companion JSON.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = analyze(plan, args.output_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown(report), encoding="utf-8")
    args.report.with_suffix(".json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        f"{report['completed']}/{report['expected']} results; {len(report['missing'])} missing"
    )


if __name__ == "__main__":
    main()
