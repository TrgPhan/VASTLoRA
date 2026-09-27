"""Re-evaluate an exported development adapter without restarting client training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

import run_kaggle_3b as shared
from week9_artifacts import validate_milestone
from riftlora.lowrank import CompactSVD
from riftlora.scale.generation import load_instruction_data
from riftlora.scale.peft_bridge import load_compact_adapter_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint_dir / "adapter.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("kind") != "evaluation_adapter_only":
        raise ValueError("not a development evaluation adapter")
    config = checkpoint["config"]
    shared._validate_config(config, checkpoint["method"])
    record, expected_identity = validate_milestone(args.checkpoint_dir, config=config,
                                                  method=checkpoint["method"], seed=checkpoint["seed"])
    if args.dry_run:
        print(json.dumps({"valid": True, "method": checkpoint["method"], "seed": checkpoint["seed"],
                          "measured_returns": record["measured_returns"], "model": config["model"]["name"]}))
        return
    if args.output_dir is None or args.output_dir.resolve() == args.checkpoint_dir.resolve():
        parser.error("re-evaluation needs a separate --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data, _ = load_instruction_data(config)
    ds = config["dataset"]
    evaluation = data[ds["eval_split"]].shuffle(seed=ds["eval_shuffle_seed"])
    offset = ds.get("eval_offset", 0)
    evaluation = evaluation.select(range(offset, offset + ds["eval_examples"]))
    if evaluation["source_id"] != expected_identity["source_id"] or evaluation["response"] != expected_identity["reference"]:
        raise ValueError("checkpoint data identity changed")
    tokenizer, model = shared._load_model(config)
    state = {name: CompactSVD(**value) for name, value in checkpoint["adapter"].items()}
    load_compact_adapter_state(model, state, active_rank=config["experiment"]["server_max_rank"],
                               initialize_free_directions=False)
    metrics, details = shared.evaluate_task(model, tokenizer, evaluation, dataset_config=ds,
                                           max_length=config["model"]["max_length"],
                                           batch_size=config["experiment"]["eval_batch_size"])
    pd.DataFrame(details).to_csv(args.output_dir / "eval_details.csv", index=False)
    result = dict(metrics=metrics, checkpoint=str(args.checkpoint_dir.resolve()),
                  git_commit=shared._git_commit(), git_worktree_dirty=shared._git_worktree_dirty(),
                  purpose="checkpoint re-evaluation, not new training or a new seed")
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
