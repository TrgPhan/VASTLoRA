from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from riftlora.lora import (
    add_dense_innovation,
    inject_diagnostic_lora,
    set_server_adapter_state,
)
from riftlora.lowrank import LowRankMatrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay one Week-3 diagnostic update")
    parser.add_argument("--csv", type=Path, default=ROOT / "outputs/week4/week3_diagnostics.csv")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--update-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.csv)
    selected = frame[(frame["run_id"] == args.run_id) & (frame["update_id"] == args.update_id)]
    if len(selected) != 1:
        raise ValueError(f"expected one row, found {len(selected)}")
    row = selected.iloc[0]
    artifact_path = args.csv.parent / str(row["update_artifact_id"]).split("#", 1)[0]
    bundle = torch.load(artifact_path, map_location="cpu", weights_only=False)
    artifact = bundle["updates"][args.update_id]
    config = bundle["config"]
    device = torch.device(args.device)

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"]["name"],
        attn_implementation="eager",
    ).to(device)
    inject_diagnostic_lora(model, target_suffixes=tuple(config["model"]["target_suffixes"]))
    current_state = bundle["snapshots"][int(artifact["current_version"])]
    innovations = {
        name: LowRankMatrix(value["left"].to(device), value["right"].to(device))
        for name, value in artifact["innovation"].items()
    }
    eval_batch = {name: value.to(device) for name, value in bundle["eval_batch"].items()}
    set_server_adapter_state(model, current_state)
    current_loss = _loss(model, eval_batch)
    candidate = add_dense_innovation(
        current_state,
        innovations,
        weight=float(row["server_update_weight"]),
    )
    set_server_adapter_state(model, candidate)
    candidate_loss = _loss(model, eval_batch)
    utility = current_loss - candidate_loss
    result = {
        "run_id": args.run_id,
        "update_id": args.update_id,
        "logged_utility": float(row["raw_update_utility"]),
        "replayed_utility": utility,
        "absolute_error": abs(utility - float(row["raw_update_utility"])),
        "passed": abs(utility - float(row["raw_update_utility"])) <= args.tolerance,
    }
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("replayed utility does not match the logged value")


def _loss(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> float:
    model.eval()
    with torch.inference_mode():
        return float(model(**batch).loss.item())


if __name__ == "__main__":
    main()

