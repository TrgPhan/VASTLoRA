from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def spearman_summary(frame: pd.DataFrame, x: str, y: str) -> dict[str, float]:
    result = spearmanr(frame[x], frame[y], nan_policy="omit")
    return {"rho": float(result.statistic), "p_value": float(result.pvalue), "n": len(frame)}


def partial_spearman(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    controls: Sequence[str],
) -> float:
    values = frame[[x, y, *controls]].dropna()
    if len(values) < len(controls) + 4:
        return math.nan
    design = np.column_stack(
        [np.ones(len(values)), *[rankdata(values[column]) for column in controls]]
    )
    x_rank = rankdata(values[x])
    y_rank = rankdata(values[y])
    x_residual = x_rank - design @ np.linalg.lstsq(design, x_rank, rcond=None)[0]
    y_residual = y_rank - design @ np.linalg.lstsq(design, y_rank, rcond=None)[0]
    return float(pearsonr(x_residual, y_residual).statistic)


def bootstrap_ci(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    seed: int = 2026,
    samples: int = 2000,
    group_column: str | None = "run_id",
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    groups = list(frame[group_column].unique()) if group_column and group_column in frame else []
    for _ in range(samples):
        if groups:
            pieces = []
            for group in groups:
                part = frame[frame[group_column] == group]
                positions = rng.integers(0, len(part), len(part))
                pieces.append(part.iloc[positions])
            sample = pd.concat(pieces, ignore_index=True)
        else:
            positions = rng.integers(0, len(frame), len(frame))
            sample = frame.iloc[positions]
        value = statistic(sample)
        if np.isfinite(value):
            estimates.append(value)
    if not estimates:
        return math.nan, math.nan
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def grouped_regression_r2(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    target: str = "raw_update_utility",
    group_column: str = "run_id",
) -> float:
    predictions, observed = _grouped_predictions(
        frame,
        features,
        target,
        group_column,
        classifier=False,
    )
    if len(observed) < 2:
        return math.nan
    return float(r2_score(observed, predictions))


def grouped_harmful_auc(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    group_column: str = "run_id",
) -> float:
    work = frame.copy()
    work["harmful"] = (work["raw_update_utility"] < 0).astype(int)
    predictions, observed = _grouped_predictions(
        work,
        features,
        "harmful",
        group_column,
        classifier=True,
    )
    if len(set(observed)) < 2:
        return math.nan
    return float(roc_auc_score(observed, predictions))


def analyze_scope(frame: pd.DataFrame) -> dict[str, Any]:
    partial = partial_spearman(
        frame,
        x="rho_two_sided",
        y="raw_update_utility",
        controls=("tau",),
    )
    partial_ci = bootstrap_ci(
        frame,
        lambda sample: partial_spearman(
            sample,
            x="rho_two_sided",
            y="raw_update_utility",
            controls=("tau",),
        ),
    )
    r2_tau = grouped_regression_r2(frame, ("tau",))
    r2_both = grouped_regression_r2(frame, ("tau", "rho_two_sided"))
    auc_tau = grouped_harmful_auc(frame, ("tau",))
    auc_rho = grouped_harmful_auc(frame, ("rho_two_sided",))
    auc_both = grouped_harmful_auc(frame, ("tau", "rho_two_sided"))
    vast_delta = float((frame["vast_update_utility"] - frame["freshness_update_utility"]).mean())
    vast_delta_ci = bootstrap_ci(
        frame,
        lambda sample: float(
            (sample["vast_update_utility"] - sample["freshness_update_utility"]).mean()
        ),
    )
    seed_partials = {
        str(seed): partial_spearman(
            part,
            x="rho_two_sided",
            y="raw_update_utility",
            controls=("tau",),
        )
        for seed, part in frame.groupby("seed")
    }
    return {
        "rows": len(frame),
        "runs": int(frame["run_id"].nunique()),
        "harmful_rate": float((frame["raw_update_utility"] < 0).mean()),
        "utility_vs_tau": spearman_summary(frame, "tau", "raw_update_utility"),
        "utility_vs_rho": spearman_summary(frame, "rho_two_sided", "raw_update_utility"),
        "partial_spearman_rho_utility_given_tau": partial,
        "partial_spearman_ci95": list(partial_ci),
        "partial_spearman_by_seed": seed_partials,
        "positive_seed_fraction": float(
            np.mean([value > 0 for value in seed_partials.values() if np.isfinite(value)])
        ),
        "cv_r2_tau": r2_tau,
        "cv_r2_tau_rho": r2_both,
        "cv_r2_gain": r2_both - r2_tau,
        "harmful_auc_tau": auc_tau,
        "harmful_auc_rho": auc_rho,
        "harmful_auc_tau_rho": auc_both,
        "harmful_auc_gain": auc_both - auc_tau,
        "raw_mean_utility": float(frame["raw_update_utility"].mean()),
        "freshness_mean_utility": float(frame["freshness_update_utility"].mean()),
        "vast_mean_utility": float(frame["vast_update_utility"].mean()),
        "raw_harmful_rate": float((frame["raw_update_utility"] < 0).mean()),
        "freshness_harmful_rate": float((frame["freshness_update_utility"] < 0).mean()),
        "vast_harmful_rate": float((frame["vast_update_utility"] < 0).mean()),
        "vast_minus_freshness_mean_utility": vast_delta,
        "vast_minus_freshness_ci95": list(vast_delta_ci),
    }


def matched_tau_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["tau_band"] = pd.cut(
        work["tau"],
        bins=[-0.5, 2.5, 7.5, np.inf],
        labels=["0-2", "3-7", "8+"],
    )
    rows = []
    for band, part in work.groupby("tau_band", observed=True):
        if len(part) < 8:
            continue
        median = part["rho_two_sided"].median()
        low = part[part["rho_two_sided"] <= median]
        high = part[part["rho_two_sided"] > median]
        rows.append(
            {
                "tau_band": str(band),
                "n": len(part),
                "spearman_rho_utility": spearman_summary(
                    part, "rho_two_sided", "raw_update_utility"
                )["rho"],
                "high_minus_low_utility": float(
                    high["raw_update_utility"].mean() - low["raw_update_utility"].mean()
                ),
                "high_rho_harmful_rate": float((high["raw_update_utility"] < 0).mean()),
                "low_rho_harmful_rate": float((low["raw_update_utility"] < 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def decide_gate(scopes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    decisions = {name: _scope_passes_gate(values) for name, values in scopes.items()}
    primary = decisions.get("iid_homogeneous", False)
    heterogeneous = decisions.get("iid_heterogeneous", False)
    conditional = decisions.get("noniid_high_staleness", False)
    if primary and heterogeneous:
        decision = "GO"
        reason = "rho adds robust signal in IID controls and survives rank heterogeneity"
    elif primary:
        decision = "CONDITIONAL GO"
        reason = "signal passes the homogeneous IID gate but is not robust to rank heterogeneity"
    elif conditional:
        decision = "CONDITIONAL GO"
        reason = "signal is supported only in the strong non-IID/high-staleness regime"
    else:
        decision = "NO-GO"
        reason = "rho does not add robust predictive and transport value beyond staleness"
    return {
        "decision": decision,
        "reason": reason,
        "scope_pass": decisions,
        "thresholds": {
            "partial_spearman_min": 0.10,
            "partial_ci_lower_min": 0.0,
            "positive_seed_fraction_min": 2 / 3,
            "predictive_gain_min_either": 0.02,
            "transport_mean_delta_min": 0.0,
            "transport_harmful_rate_not_worse": True,
        },
    }


def _scope_passes_gate(values: dict[str, Any]) -> bool:
    predictive_gain = max(values["cv_r2_gain"], values["harmful_auc_gain"])
    return bool(
        values["partial_spearman_rho_utility_given_tau"] >= 0.10
        and values["partial_spearman_ci95"][0] > 0.0
        and values["positive_seed_fraction"] >= 2 / 3
        and predictive_gain >= 0.02
        and values["vast_minus_freshness_mean_utility"] > 0.0
        and values["vast_harmful_rate"] <= values["freshness_harmful_rate"]
    )


def _grouped_predictions(
    frame: pd.DataFrame,
    features: Sequence[str],
    target: str,
    group_column: str,
    *,
    classifier: bool,
) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    groups = list(frame[group_column].unique())
    if len(groups) < 2:
        return np.array([]), np.array([])
    for held_out in groups:
        train = frame[frame[group_column] != held_out]
        test = frame[frame[group_column] == held_out]
        if classifier and (train[target].nunique() < 2 or test[target].nunique() < 2):
            continue
        if classifier:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=2000, random_state=0),
            )
            model.fit(train[list(features)], train[target])
            prediction = model.predict_proba(test[list(features)])[:, 1]
        else:
            model = make_pipeline(StandardScaler(), LinearRegression())
            model.fit(train[list(features)], train[target])
            prediction = model.predict(test[list(features)])
        predictions.append(np.asarray(prediction))
        observed.append(test[target].to_numpy())
    if not predictions:
        return np.array([]), np.array([])
    return np.concatenate(predictions), np.concatenate(observed)
