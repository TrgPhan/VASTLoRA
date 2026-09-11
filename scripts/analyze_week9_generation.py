"""Completeness, provenance, token-NLL and paired confirmation checks for Week 9."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_week9_generation import specs
from week9_artifacts import validate_run


def paired_interval(values):
    x = np.asarray(values, dtype=float)
    if len(x) < 2 or not np.isfinite(x).all():
        return None, None, None
    mean = float(x.mean())
    radius = float(t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))
    return mean, mean - radius, mean + radius


def analyze(root, target="rift_core"):
    matrix = json.loads((root / "matrix.json").read_text())
    if target not in matrix["methods"]:
        raise ValueError("target must be declared in the matrix")
    rows, issues, missing, paired_data, baseline_values = [], [], [], {}, {}
    evaluation_identity = None
    for method, seed, config, path in specs(matrix, root):
        key = f"{config['experiment']['regime_name']}/{method}/{seed}"
        if not path.exists():
            missing.append(key)
            continue
        try:
            payload, identity = validate_run(path, config=config, method=method, seed=seed, matrix=matrix)
            m = payload["metrics"]
            if payload.get("git_worktree_dirty") is not False:
                issues.append(f"{key}: dirty or unverified checkout")
            exp = config["experiment"]
            pair_key = (exp["regime_name"], seed)
            if pair_key in paired_data and paired_data[pair_key] != identity:
                raise ValueError("data/schedule is not matched across methods for this seed")
            paired_data[pair_key] = identity
            if evaluation_identity is not None and evaluation_identity != identity["references"]:
                raise ValueError("fixed evaluation set changed across seeds/regimes")
            evaluation_identity = identity["references"]
            if pair_key in baseline_values and not math.isclose(
                    baseline_values[pair_key], m["baseline_token_nll"], rel_tol=1e-4, abs_tol=1e-6):
                raise ValueError("starting backbone NLL differs across paired methods")
            baseline_values[pair_key] = m["baseline_token_nll"]
            rows.append({"regime": exp["regime_name"], "method": method, "seed": seed,
                         "token_nll": m["final_token_nll"], "perplexity": m["final_perplexity"],
                         "rouge_l": m["final_rouge_l"], "exact_match": m["final_exact_match"],
                         "backbone_nll_change": m["final_token_nll"] - m["baseline_token_nll"],
                         "harmful": m["harmful_update_rate"],
                         "late_harmful": m["late_harmful_update_rate"] if m["late_event_count"] else None,
                         "late_events": m["late_event_count"], "acceptance": m["acceptance_rate"],
                         "runtime_seconds": m["runtime_seconds"], "peak_vram_gib": m["peak_cuda_memory_gib"],
                         "generation_limit_rate": m["final_generation_limit_rate"],
                         "git_commit": payload.get("git_commit")})
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(f"{key}: {error}")
    pairs = []
    gates = matrix["gates"]
    frame = pd.DataFrame(rows)
    if rows:
        if len(set(frame.git_commit)) != 1 or not all(isinstance(v, str) and len(v) == 40
                and all(c in "0123456789abcdef" for c in v) for v in frame.git_commit):
            issues.append("mixed or invalid implementation commits")
        for regime in matrix["regimes"]:
            group = frame[frame.regime == regime["name"]]
            for baseline in gates["baselines"]:
                left = group[group.method == target].set_index("seed")
                right = group[group.method == baseline].set_index("seed")
                seeds = sorted(set(left.index) & set(right.index))
                mean, lower, upper = paired_interval([left.loc[s, "token_nll"] - right.loc[s, "token_nll"] for s in seeds])
                pairs.append({"regime": regime["name"], "baseline": baseline, "paired_seeds": len(seeds),
                              "target_minus_baseline_nll": mean, "ci95_low": lower, "ci95_high": upper,
                              "nll_noninferior": upper is not None and upper <= gates["maximum_token_nll_increase"]})
    status = "INCOMPLETE_OR_UNVERIFIED"
    if matrix["phase"] == "smoke":
        status = "SMOKE_ONLY"
    elif matrix["phase"] == "development":
        status = "DEVELOPMENT_ONLY"
    elif target != matrix.get("primary_target", "rift_core"):
        status = "EXPLORATORY_TARGET_ONLY"
    elif not missing and not issues and rows:
        coverage = (frame.late_events >= gates["minimum_late_events"]).all()
        progress = (frame[frame.method == target].acceptance >= gates["minimum_acceptance_rate"]).all()
        paired = pairs and all(p["paired_seeds"] >= gates["minimum_paired_seeds"] for p in pairs)
        if not coverage or not progress or not paired:
            status = "INCONCLUSIVE_COVERAGE_OR_PROGRESS"
        elif all(p["nll_noninferior"] for p in pairs):
            status = "PASS_WEEK9_NLL_GATE_ONLY"
        elif any(p["ci95_low"] > gates["maximum_token_nll_increase"] for p in pairs):
            status = "NO_GO_NLL"
        else:
            status = "INCONCLUSIVE_NLL"
    return {"status": status, "target": target, "phase": matrix["phase"], "rows": rows,
            "pairs": pairs, "missing": missing, "issues": issues,
            "warnings": ["Generation hit the token cap: inspect predictions before interpreting sequence metrics"]
                        if rows and (frame.generation_limit_rate > 0).any() else [],
            "scope": "Week 9 NLL noninferiority only, not global thesis GO"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--target", default="rift_core", choices=["rift", "rift_diag", "rift_core"])
    args = parser.parse_args()
    report = analyze(args.input_dir, args.target)
    output = args.input_dir / f"analysis_{args.target}"
    output.mkdir(exist_ok=True)
    (output / "verdict.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    text = ["# Week 9 generative results", "", report["status"], "", report["scope"], ""]
    if report["rows"]:
        frame = pd.DataFrame(report["rows"])
        frame.to_csv(output / "runs.csv", index=False)
        summary = frame.groupby(["regime", "method"]).agg(
            seeds=("seed", "count"), token_nll=("token_nll", "mean"), nll_sd=("token_nll", "std"),
            perplexity=("perplexity", "mean"), rouge_l=("rouge_l", "mean"),
            harmful=("harmful", "mean"), late_harmful=("late_harmful", "mean"),
            minimum_late_events=("late_events", "min"), acceptance=("acceptance", "mean"),
            runtime_seconds=("runtime_seconds", "mean"), peak_vram_gib=("peak_vram_gib", "max"),
            generation_limit_rate=("generation_limit_rate", "mean"),
        ).reset_index()
        text.extend([summary.to_markdown(index=False), ""])
        text.extend(["## Paired NLL intervals", "", pd.DataFrame(report["pairs"]).to_markdown(index=False), ""])
    text.extend(["## Integrity", "", f"Missing: {len(report['missing'])}; issues: {len(report['issues'])}", "",
                 *report["issues"], *report["warnings"]])
    (output / "results.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
