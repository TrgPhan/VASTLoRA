from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_METHODS = {"freshness", "vast", "mtip", "mtip_adaptive"}


def summarize_results(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    records = []
    for path in sorted(input_dir.glob("*_seed*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            {
                "method": payload["method"],
                "seed": payload["seed"],
                "model": payload["model"],
                "git_commit": payload["git_commit"],
                **payload["metrics"],
            }
        )
    if not records:
        raise ValueError(f"no result.json files found under {input_dir}")

    frame = pd.DataFrame(records).sort_values(["seed", "method"]).reset_index(drop=True)
    comparisons = _paired_comparisons(frame)
    missing = sorted(EXPECTED_METHODS - set(frame["method"]))
    verdict = _build_verdict(frame, comparisons, missing)
    return frame, comparisons, verdict


def render_verdict_markdown(verdict: dict[str, Any]) -> str:
    lines = [f"## 3B verdict: {verdict['status']}", "", verdict["reason"]]
    if "adaptive_accuracy_gain_vs_freshness_pp" in verdict:
        lines.extend(
            [
                "",
                f"- Adaptive MTIP accuracy gain: {verdict['adaptive_accuracy_gain_vs_freshness_pp']:+.3f} pp",
                f"- Adaptive MTIP NLL gain: {verdict['adaptive_nll_gain_vs_freshness']:+.6f}",
                f"- Accuracy wins: {verdict['adaptive_accuracy_wins']}/{verdict['seed_count']}",
                f"- Best mean-accuracy method: `{verdict['best_method_by_mean_accuracy']}`",
                "",
                "`PILOT_GO` is provisional. A full `GO` requires at least three seeds.",
            ]
        )
    return "\n".join(lines) + "\n"


def write_summary(input_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary, comparisons, verdict = summarize_results(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "method_summary.csv", index=False)
    comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)
    (output_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "verdict.md").write_text(
        render_verdict_markdown(verdict), encoding="utf-8"
    )
    return summary, verdict


def _paired_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed, group in frame.groupby("seed"):
        by_method = group.set_index("method")
        if "freshness" not in by_method.index:
            continue
        baseline = by_method.loc["freshness"]
        for method in ("vast", "mtip", "mtip_adaptive"):
            if method not in by_method.index:
                continue
            candidate = by_method.loc[method]
            rows.append(
                {
                    "seed": int(seed),
                    "method": method,
                    "accuracy_gain_vs_freshness_pp": 100.0
                    * (candidate["final_accuracy"] - baseline["final_accuracy"]),
                    "nll_gain_vs_freshness": baseline["final_nll"] - candidate["final_nll"],
                    "runtime_ratio_vs_freshness": candidate["runtime_seconds"]
                    / baseline["runtime_seconds"],
                    "memory_delta_vs_freshness_gib": candidate["peak_cuda_memory_gib"]
                    - baseline["peak_cuda_memory_gib"],
                }
            )
    return pd.DataFrame(rows)


def _build_verdict(
    frame: pd.DataFrame,
    comparisons: pd.DataFrame,
    missing: list[str],
) -> dict[str, Any]:
    if missing:
        return {
            "status": "INCOMPLETE",
            "reason": f"Missing methods: {', '.join(missing)}",
            "missing_methods": missing,
            "seed_count": int(frame["seed"].nunique()),
        }

    adaptive = comparisons[comparisons["method"] == "mtip_adaptive"]
    mean_accuracy_gain = float(adaptive["accuracy_gain_vs_freshness_pp"].mean())
    mean_nll_gain = float(adaptive["nll_gain_vs_freshness"].mean())
    wins = int((adaptive["accuracy_gain_vs_freshness_pp"] > 0).sum())
    seed_count = int(adaptive["seed"].nunique())
    best_method = (
        frame.groupby("method")["final_accuracy"].mean().sort_values(ascending=False).index[0]
    )

    if mean_accuracy_gain >= 0.5 and mean_nll_gain >= 0.0:
        status = "PILOT_GO" if seed_count < 3 else "GO"
        reason = "Adaptive MTIP improves both accuracy and NLL over freshness."
    elif mean_accuracy_gain <= -0.5 and mean_nll_gain < 0.0:
        status = "NO_GO"
        reason = "Adaptive MTIP is materially worse than freshness on accuracy and NLL."
    else:
        status = "INCONCLUSIVE"
        reason = "Accuracy and NLL do not provide a consistent margin over freshness."

    return {
        "status": status,
        "reason": reason,
        "best_method_by_mean_accuracy": str(best_method),
        "adaptive_accuracy_gain_vs_freshness_pp": mean_accuracy_gain,
        "adaptive_nll_gain_vs_freshness": mean_nll_gain,
        "adaptive_accuracy_wins": wins,
        "seed_count": seed_count,
        "gate": {
            "minimum_accuracy_gain_pp": 0.5,
            "requires_nonnegative_nll_gain": True,
            "minimum_seeds_for_full_go": 3,
        },
    }
