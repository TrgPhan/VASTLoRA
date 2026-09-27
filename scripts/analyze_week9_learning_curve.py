"""Read completed development milestones, including those in interrupted jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from run_week9_generation import specs
from week9_artifacts import validate_milestone


def analyze(root):
    matrix = json.loads((root / "matrix.json").read_text())
    if matrix["phase"] not in {"development", "smoke"}:
        raise ValueError("learning-curve analysis is development only")
    rows, missing, issues, review_rows, review_keys = [], [], [], [], []
    identity = None
    for method, seed, config, path in specs(matrix, root):
        for budget in config["experiment"].get("generation_eval_returns", []):
            directory = path.parent / "milestones" / f"returns_{budget:04d}"
            key = f"{config['experiment']['regime_name']}/{method}/{seed}/{budget}"
            if not (directory / "metrics.json").exists():
                missing.append(key)
                continue
            try:
                record, current = validate_milestone(directory, config=config, method=method, seed=seed)
                if identity is not None and identity != current:
                    raise ValueError("evaluation identity differs across milestones/methods/seeds")
                identity = current
                row = {k: record[k] for k in ("method", "seed", "regime", "measured_returns", "prompt_format",
                                              "git_commit", "git_worktree_dirty", "elapsed_seconds")}
                row.update({k: record["metrics"][k] for k in ("nll", "rouge_l", "rouge_l_precision", "rouge_l_recall",
                                                              "exact_match", "mean_generated_tokens", "generation_limit_rate")})
                row.update(record["work"])
                row.update(record["safety"])
                rows.append(row)
                # Fixed first 32 evaluation questions at every milestone/method, never chosen by score.
                details = pd.read_csv(directory / "eval_details.csv", keep_default_na=False)
                for item in details.head(32).to_dict("records"):
                    review_id = hashlib.sha256(json.dumps([matrix["name"], key, item["source_id"],
                                                          item["reference"], item["prediction"]]).encode()).hexdigest()[:20]
                    review_rows.append(dict(review_id=review_id, source_id=item["source_id"],
                                            instruction=item["instruction"], context=item["context"],
                                            reference=item["reference"], prediction=item["prediction"],
                                            correctness="", notes=""))
                    review_keys.append(dict(review_id=review_id, method=method, seed=seed, measured_returns=budget))
            except (OSError, ValueError, KeyError, TypeError) as error:
                issues.append(f"{key}: {error}")
    commits = {r["git_commit"] for r in rows}
    if len(commits) > 1:
        issues.append("mixed implementation commits; do not pool results")
    if any(r["git_worktree_dirty"] is not False for r in rows):
        issues.append("dirty development checkout: diagnostic only, not frozen evidence")
    return dict(rows=rows, missing=missing, issues=issues, review_rows=review_rows, review_keys=review_keys,
                status="DEVELOPMENT_ONLY", phase=matrix["phase"], matrix_name=matrix["name"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.input_dir)
    output = args.input_dir / "learning_curve_analysis"
    output.mkdir(exist_ok=True)
    frame = pd.DataFrame(report.pop("rows"))
    frame.to_csv(output / "runs.csv", index=False)
    if frame.empty:
        pd.DataFrame(columns=["method", "measured_returns", "seeds", "nll", "rouge_l"]).to_csv(output / "summary.csv", index=False)
        (output / "learning_curves.png").unlink(missing_ok=True)
    review_rows, review_keys = report.pop("review_rows"), report.pop("review_keys")
    if review_rows:
        review = pd.DataFrame(review_rows).sort_values("review_id").set_index("review_id")
        destination = output / "correctness_template.csv"
        if destination.exists():
            previous = pd.read_csv(destination, keep_default_na=False).set_index("review_id")
            for column in ("correctness", "notes"):
                review[column] = previous[column].reindex(review.index).fillna("")
        review.to_csv(destination)
        pd.DataFrame(review_keys).to_csv(output / "correctness_key.csv", index=False)
    text = ["# Week 9 development learning curves", "", report["status"], "",
            f"Missing milestones: {len(report['missing'])}. Issues: {len(report['issues'])}.", "",
            "Each run is one continuous trajectory; checkpoints are not independent seeds.", ""]
    if not frame.empty:
        summary = frame.groupby(["prompt_format", "method", "measured_returns"]).agg(
            seeds=("seed", "count"), nll=("nll", "mean"), nll_sd=("nll", "std"),
            rouge_l=("rouge_l", "mean"), precision=("rouge_l_precision", "mean"), recall=("rouge_l_recall", "mean"),
            exact_match=("exact_match", "mean"), generated_tokens=("mean_generated_tokens", "mean"),
            limit_rate=("generation_limit_rate", "mean"), samples=("client_sample_presentations", "mean"),
            unique_samples=("client_unique_examples", "mean"), response_tokens=("client_response_tokens", "mean"),
            harmful=("harmful_update_rate", "mean"), late_harmful=("late_harmful_update_rate", "mean"),
            acceptance=("acceptance_rate", "mean"),
        ).reset_index()
        summary.to_csv(output / "summary.csv", index=False)
        text.extend([summary.to_markdown(index=False), ""])
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for (_, method), group in summary.groupby(["prompt_format", "method"]):
            for ax, metric in zip(axes, ("nll", "rouge_l", "generated_tokens")):
                ax.plot(group.measured_returns, group[metric], marker="o", label=method)
                ax.set(xlabel="Measured returns", ylabel=metric)
                ax.grid(alpha=0.25)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "learning_curves.png", dpi=160)
        plt.close(fig)
    text.extend(["## Integrity", "", *report["issues"], "", "## Missing", "", *report["missing"]])
    (output / "results.md").write_text("\n".join(text), encoding="utf-8")
    (output / "status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
