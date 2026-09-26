"""Frozen Week 8 protocol overlays for isolated FedLoRA baseline suites."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("flexlora", "florist", "florg")
SUITE_VERSION = "week8-spectral-v1"
SUITES = {
    "spectral_only": (METHODS, SUITE_VERSION),
    "factor_only": (("fedavg_lora", "ffa_lora"), "week8-factor-v2"),
}


def build_matrix(*, smoke: bool = False, suite: str = "spectral_only") -> dict:
    if suite not in SUITES:
        raise ValueError(f"Unknown method suite: {suite}")
    matrix = json.loads((ROOT / "configs/rift_core_heldout_confirmation_matrix.json").read_text(encoding="utf-8"))
    matrix["name"] = SUITE_VERSION + ("-smoke" if smoke else "-confirmation")
    matrix["methods"] = list(METHODS)
    matrix["description"] = "FlexLoRA/FLoRIST/FLoRG operators under the matched immediate-async Week 8 protocol."
    matrix["experiment"]["florist_energy"] = 0.9
    matrix["experiment"]["spectral_suite_version"] = SUITE_VERSION
    matrix["notes"] = [
        "Only flexlora, florist and florg run. Four tasks, seeds 6101-6106, same evaluation slices as Week 8.",
        "FlexLoRA/FLoRIST interpolate complete returned client and current server states; this is an explicit async adaptation.",
        "FlexLoRA uses exact compact SVD; FLoRIST uses factor SVDs, intermediate SVD, squared-energy tau=0.9 and a logged shared rank cap.",
        "FlexLoRA/FLoRIST preserve 8 freshness warmup returns; free factor directions are initialized only at base version zero, then zero-padded.",
        "FLoRG trains a single matrix and uses Gram/rectangular Procrustes aggregation from the first return. Warmup only excludes those events from safety metrics.",
        "FLoRG heterogeneous dispatch and fixed rank budget are explicit extensions; rectangular projection cannot preserve a higher-rank Gram exactly.",
        "No server calibration optimization is used by these three methods. Monitor data only measures harmful updates.",
        "Same model/data revisions and held-out offsets as the base matrix; do not tune on these confirmation seeds.",
    ]
    if suite == "factor_only":
        methods, version = SUITES[suite]
        matrix["name"] = version + ("-smoke" if smoke else "-confirmation")
        matrix["methods"] = list(methods)
        matrix["description"] = "Persistent-factor FedAvg/FFA under the matched immediate-async Week 8 protocol."
        matrix["experiment"].pop("florist_energy", None)
        matrix["experiment"].pop("spectral_suite_version", None)
        matrix["experiment"]["factor_implementation"] = "persistent_factors_v2"
        matrix["notes"] = [
            "Only fedavg_lora and ffa_lora run; four tasks and seeds 6101-6106 are unchanged.",
            "FedAvg retains actual A/B snapshots; FFA preserves shared immutable A0 and aggregates only B.",
            "Native method during unmeasured warmup; no freshness warmup or SVD factor reload.",
            "Immediate-async interpolation and heterogeneous prefix ranks are explicit paper adaptations, no DP.",
            "Separate output root; legacy factor results must not be reused or merged as corrected baselines.",
        ]
    if smoke:
        matrix["tasks"] = [matrix["tasks"][0]]
        matrix["seeds"] = [6101]
        matrix["tasks"][0].update(max_train_examples=32, eval_examples=8)
        matrix["experiment"].update(warmup_returns=2, collected_returns=4,
                                    calibration_gradient_examples=2,
                                    calibration_gate_examples=4, monitor_examples=4)
        matrix["notes"].append("Smoke is diagnostic only; never merge with confirmation.")
    return matrix


def job_list(matrix: dict) -> list[tuple[str, str, str, int]]:
    candidates = [methods for methods, version in SUITES.values()
                  if matrix["name"] in {version + "-smoke", version + "-confirmation"}]
    if len(candidates) != 1 or matrix["methods"] != list(candidates[0]):
        raise ValueError("Method selection must match the declared isolated suite")
    return [(t["name"], r["name"], m, int(s)) for t in matrix["tasks"]
            for r in matrix["regimes"] for m in candidates[0] for s in matrix["seeds"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--suite", choices=tuple(SUITES), default="spectral_only")
    args = parser.parse_args()
    matrix = build_matrix(smoke=args.smoke, suite=args.suite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"matrix": str(args.output), "methods": matrix["methods"], "jobs": len(job_list(matrix))}))
