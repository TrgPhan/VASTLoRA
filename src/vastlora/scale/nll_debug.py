from __future__ import annotations

from dataclasses import dataclass
import math
import json
from pathlib import Path
from typing import Any

import pandas as pd


DETAIL_COLUMNS = {
    "eval_index",
    "text",
    "true_label",
    "predicted_label",
    "is_correct",
    "true_nll",
    "wrong_nll",
    "nll_margin",
    "true_probability",
    "prediction_confidence",
}


@dataclass(frozen=True)
class NllDebugOutputs:
    output_dir: Path
    run_summary: pd.DataFrame
    label_summary: pd.DataFrame
    paired_summary: pd.DataFrame
    worst_samples: pd.DataFrame
    calibration: pd.DataFrame
    event_summary: pd.DataFrame
    missing_runs: list[str]


def write_nll_debug(input_dir: Path, output_dir: Path | None = None) -> NllDebugOutputs:
    output = output_dir or input_dir / "summary" / "nll_debug"
    output.mkdir(parents=True, exist_ok=True)

    details, missing = _load_details(input_dir)
    events = _load_events(input_dir)
    if not details:
        empty = pd.DataFrame()
        markdown = render_nll_debug_markdown(
            empty,
            empty,
            empty,
            empty,
            empty,
            missing_runs=missing,
        )
        (output / "nll_debug.md").write_text(markdown, encoding="utf-8")
        return NllDebugOutputs(output, empty, empty, empty, empty, empty, empty, missing)

    frame = pd.concat(details, ignore_index=True)
    run_summary = _run_summary(frame)
    label_summary = _label_summary(frame)
    paired_summary, worst_samples = _paired_against_freshness(frame)
    calibration = _calibration_summary(frame)
    event_summary = _event_summary(events)

    run_summary.to_csv(output / "nll_run_summary.csv", index=False)
    label_summary.to_csv(output / "nll_label_summary.csv", index=False)
    paired_summary.to_csv(output / "nll_paired_vs_freshness.csv", index=False)
    worst_samples.to_csv(output / "nll_worst_samples.csv", index=False)
    calibration.to_csv(output / "nll_calibration_buckets.csv", index=False)
    event_summary.to_csv(output / "event_debug_summary.csv", index=False)
    markdown = render_nll_debug_markdown(
        run_summary,
        label_summary,
        paired_summary,
        worst_samples,
        event_summary,
        missing_runs=missing,
    )
    (output / "nll_debug.md").write_text(markdown, encoding="utf-8")
    return NllDebugOutputs(
        output,
        run_summary,
        label_summary,
        paired_summary,
        worst_samples,
        calibration,
        event_summary,
        missing,
    )


def render_nll_debug_markdown(
    run_summary: pd.DataFrame,
    label_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
    worst_samples: pd.DataFrame,
    event_summary: pd.DataFrame,
    *,
    missing_runs: list[str],
) -> str:
    lines = ["## NLL forensic debug", ""]
    if missing_runs:
        lines.extend(
            [
                "Some runs do not contain per-example NLL artifacts.",
                "",
                "Missing `final_eval_details.csv`: " + ", ".join(missing_runs),
                "",
                "Re-run the notebook from the updated commit to generate sample-level NLL diagnostics.",
            ]
        )
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "This debug pass compares final per-example label NLL against Freshness on the same seed and validation examples.",
            "",
            "### Run-level NLL",
            _markdown_table(run_summary),
            "",
            "### Paired vs Freshness",
            _markdown_table(paired_summary),
            "",
            "### Event geometry",
            _markdown_table(event_summary),
            "",
            "### Worst NLL regressions",
            _markdown_table(worst_samples.head(12)),
        ]
    )
    if not label_summary.empty:
        lines.extend(["", "### Label split", _markdown_table(label_summary)])
    return "\n".join(lines) + "\n"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.to_csv(index=False).strip()


def _load_details(input_dir: Path) -> tuple[list[pd.DataFrame], list[str]]:
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for result_path in sorted(input_dir.glob("*_seed*/result.json")):
        run_dir = result_path.parent
        detail_path = run_dir / "final_eval_details.csv"
        if not detail_path.exists():
            missing.append(run_dir.name)
            continue
        frame = pd.read_csv(detail_path)
        missing_columns = DETAIL_COLUMNS - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{detail_path} is missing columns: {sorted(missing_columns)}"
            )
        if "binary_nll" not in frame.columns:
            frame["binary_nll"] = -frame["true_probability"].clip(lower=1e-12).map(
                math.log
            )
        for optional_nll in ("label_nll", "eos_nll"):
            if optional_nll not in frame.columns:
                frame[optional_nll] = float("nan")
        if "brier" not in frame.columns:
            frame["brier"] = (frame["prob_positive"] - frame["true_label"]) ** 2
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        frame["method"] = payload.get("variant", payload.get("method", run_dir.name))
        frame["seed"] = int(payload["seed"])
        frames.append(frame)
    return frames, missing


def _load_events(input_dir: Path) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for event_path in sorted(input_dir.glob("*_seed*/events.csv")):
        frame = pd.read_csv(event_path)
        result_path = event_path.parent / "result.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        frame["method"] = payload.get("variant", payload.get("method", event_path.parent.name))
        frame["seed"] = int(payload["seed"])
        frames.append(frame)
    return frames


def _run_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["method", "seed"], as_index=False)
    summary = grouped.agg(
        accuracy=("is_correct", "mean"),
        mean_true_nll=("true_nll", "mean"),
        mean_binary_nll=("binary_nll", "mean"),
        mean_label_nll=("label_nll", "mean"),
        mean_eos_nll=("eos_nll", "mean"),
        mean_brier=("brier", "mean"),
        median_true_nll=("true_nll", "median"),
        p90_true_nll=("true_nll", lambda values: values.quantile(0.90)),
        p95_true_nll=("true_nll", lambda values: values.quantile(0.95)),
        max_true_nll=("true_nll", "max"),
        mean_margin=("nll_margin", "mean"),
        mean_true_probability=("true_probability", "mean"),
        wrong_overconfident_rate=(
            "prediction_confidence",
            lambda values: float(
                (
                    (frame.loc[values.index, "is_correct"] == 0)
                    & (values >= 0.90)
                ).mean()
            ),
        ),
    )
    return summary.sort_values(["seed", "method"]).reset_index(drop=True)


def _label_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["method", "seed", "true_label"], as_index=False)
        .agg(
            count=("eval_index", "count"),
            accuracy=("is_correct", "mean"),
            mean_true_nll=("true_nll", "mean"),
            mean_binary_nll=("binary_nll", "mean"),
            p90_true_nll=("true_nll", lambda values: values.quantile(0.90)),
            mean_true_probability=("true_probability", "mean"),
        )
        .sort_values(["seed", "method", "true_label"])
        .reset_index(drop=True)
    )


def _paired_against_freshness(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    worst_rows: list[pd.DataFrame] = []
    for seed, seed_frame in frame.groupby("seed"):
        freshness = seed_frame[seed_frame["method"] == "freshness"]
        if freshness.empty:
            continue
        base = freshness.set_index("eval_index")
        for method, candidate in seed_frame.groupby("method"):
            if method == "freshness":
                continue
            joined = candidate.set_index("eval_index").join(
                base[
                    [
                        "true_nll",
                        "binary_nll",
                        "brier",
                        "is_correct",
                        "predicted_label",
                        "prediction_confidence",
                    ]
                ],
                rsuffix="_freshness",
                how="inner",
            )
            joined["delta_true_nll_vs_freshness"] = (
                joined["true_nll"] - joined["true_nll_freshness"]
            )
            joined["delta_binary_nll_vs_freshness"] = (
                joined["binary_nll"] - joined["binary_nll_freshness"]
            )
            joined["delta_brier_vs_freshness"] = (
                joined["brier"] - joined["brier_freshness"]
            )
            joined["candidate_fixed_freshness_error"] = (
                (joined["is_correct_freshness"] == 0) & (joined["is_correct"] == 1)
            )
            joined["candidate_broke_freshness_correct"] = (
                (joined["is_correct_freshness"] == 1) & (joined["is_correct"] == 0)
            )
            rows.append(
                {
                    "seed": int(seed),
                    "method": method,
                    "mean_delta_nll_vs_freshness": joined[
                        "delta_true_nll_vs_freshness"
                    ].mean(),
                    "mean_delta_binary_nll_vs_freshness": joined[
                        "delta_binary_nll_vs_freshness"
                    ].mean(),
                    "mean_delta_brier_vs_freshness": joined[
                        "delta_brier_vs_freshness"
                    ].mean(),
                    "median_delta_nll_vs_freshness": joined[
                        "delta_true_nll_vs_freshness"
                    ].median(),
                    "p90_delta_nll_vs_freshness": joined[
                        "delta_true_nll_vs_freshness"
                    ].quantile(0.90),
                    "samples_nll_worse_by_1": int(
                        (joined["delta_true_nll_vs_freshness"] > 1.0).sum()
                    ),
                    "samples_nll_worse_by_5": int(
                        (joined["delta_true_nll_vs_freshness"] > 5.0).sum()
                    ),
                    "fixed_freshness_errors": int(
                        joined["candidate_fixed_freshness_error"].sum()
                    ),
                    "broke_freshness_correct": int(
                        joined["candidate_broke_freshness_correct"].sum()
                    ),
                    "shared_correct_mean_delta": joined[
                        (joined["is_correct"] == 1)
                        & (joined["is_correct_freshness"] == 1)
                    ]["delta_true_nll_vs_freshness"].mean(),
                }
            )
            sample_cols = [
                "text",
                "true_label",
                "predicted_label",
                "predicted_label_freshness",
                "is_correct",
                "is_correct_freshness",
                "true_nll",
                "true_nll_freshness",
                "delta_true_nll_vs_freshness",
                "prediction_confidence",
                "prediction_confidence_freshness",
            ]
            worst = (
                joined.reset_index()
                .sort_values("delta_true_nll_vs_freshness", ascending=False)
                .head(8)
            )
            worst["seed"] = int(seed)
            worst["method"] = method
            worst_rows.append(worst[["seed", "method", "eval_index", *sample_cols]])
    paired = pd.DataFrame(rows).sort_values(["seed", "method"]).reset_index(drop=True)
    worst_samples = (
        pd.concat(worst_rows, ignore_index=True)
        if worst_rows
        else pd.DataFrame()
    )
    if not worst_samples.empty:
        worst_samples = worst_samples.sort_values(
            "delta_true_nll_vs_freshness", ascending=False
        ).reset_index(drop=True)
    return paired, worst_samples


def _calibration_summary(frame: pd.DataFrame) -> pd.DataFrame:
    buckets = pd.IntervalIndex.from_tuples(
        [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)],
        closed="right",
    )
    calibrated = frame.copy()
    calibrated["confidence_bucket"] = pd.cut(
        calibrated["prediction_confidence"], buckets
    ).astype(str)
    return (
        calibrated.groupby(["method", "seed", "confidence_bucket"], as_index=False)
        .agg(
            count=("eval_index", "count"),
            accuracy=("is_correct", "mean"),
            mean_true_nll=("true_nll", "mean"),
            mean_confidence=("prediction_confidence", "mean"),
        )
        .sort_values(["seed", "method", "confidence_bucket"])
        .reset_index(drop=True)
    )


def _event_summary(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for frame in frames:
        method = frame["method"].iloc[0]
        seed = int(frame["seed"].iloc[0])
        post = frame.iloc[4:] if len(frame) > 4 else frame
        rows.append(
            {
                "method": method,
                "seed": seed,
                "returns": len(frame),
                "mean_staleness": frame["staleness"].mean(),
                "max_staleness": int(frame["staleness"].max()),
                "post_mean_local_loss": post["local_loss"].mean(),
                "post_mean_rho": post["rho"].mean(),
                "post_p90_rho": post["rho"].quantile(0.90),
                "post_mean_residual_scale": (
                    post["residual_scale"].mean()
                    if "residual_scale" in post.columns
                    else float("nan")
                ),
                "post_mean_left_rank": post["mean_left_rank"].mean(),
                "post_mean_right_rank": post["mean_right_rank"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["seed", "method"]).reset_index(drop=True)
