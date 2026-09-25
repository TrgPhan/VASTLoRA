"""Development learning curves; evaluation must not change the training trajectory."""
from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from riftlora.scale.peft_bridge import named_peft_lora_modules, load_compact_adapter_state


class ClientWork:
    def __init__(self):
        self.examples = self.input_tokens = self.response_tokens = 0
        self.source_ids = set()

    def record(self, examples, batch):
        self.examples += len(examples)
        self.source_ids.update(int(row["source_id"]) for row in examples)
        self.input_tokens += int(batch["attention_mask"].sum())
        self.response_tokens += int(batch["labels"].ne(-100).sum())

    def snapshot(self):
        return dict(client_sample_presentations=self.examples, client_unique_examples=len(self.source_ids),
                    client_input_tokens=self.input_tokens, client_response_tokens=self.response_tokens)


@contextmanager
def isolated_evaluation(model):
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    modes = [(module, module.training) for module in model.modules()]
    factors = [(module.lora_A["default"].weight, module.lora_A["default"].weight.detach().clone(),
                module.lora_B["default"].weight, module.lora_B["default"].weight.detach().clone())
               for module in named_peft_lora_modules(model).values()]
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            with torch.no_grad():
                for a, saved_a, b, saved_b in factors:
                    a.copy_(saved_a)
                    b.copy_(saved_b)
            for module, training in modes:
                module.training = training
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)


def evaluate_server(model, server_state, *, rank, evaluate):
    with isolated_evaluation(model):
        load_compact_adapter_state(model, server_state, active_rank=rank, initialize_free_directions=False)
        return evaluate()


def write_milestone(directory, *, config, method, seed, measured_returns, total_returns,
                    server_version, server_state, metrics, details, work, elapsed_seconds,
                    git_commit, git_worktree_dirty, config_fingerprint, events):
    directory = Path(directory) / "milestones" / f"returns_{measured_returns:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = dict(kind="evaluation_adapter_only", config=config, method=method, seed=seed,
                      measured_returns=measured_returns, total_returns=total_returns,
                      server_version=server_version,
                      adapter={name: {key: getattr(value, key).detach().cpu() for key in ("u", "s", "v")}
                               for name, value in server_state.items()})
    temporary = directory / "adapter.tmp"
    torch.save(checkpoint, temporary)
    temporary.replace(directory / "adapter.pt")
    with (directory / "eval_details.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    with (directory / "events.csv").open("w", encoding="utf-8", newline="") as handle:
        keys = list(dict.fromkeys(key for row in events for key in row))
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(events)
    record = dict(method=method, seed=seed, measured_returns=measured_returns,
                  total_returns=total_returns, server_version=server_version, metrics=metrics,
                  work=work, elapsed_seconds=elapsed_seconds, config_fingerprint=config_fingerprint,
                  git_commit=git_commit, git_worktree_dirty=git_worktree_dirty,
                  phase=config["provenance"]["phase"],
                  prompt_format=config["dataset"]["prompt_format"],
                  regime=config["experiment"]["regime_name"],
                  checkpoint_kind="evaluation_adapter_only")
    record["sha256"] = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                         for name in ("adapter.pt", "eval_details.csv", "events.csv")}
    measured = [row for row in events if row["measured"]]
    late = [row for row in measured if row["staleness"] >= config["experiment"]["late_tau"]]
    record["safety"] = dict(harmful_update_rate=sum(row["harmful_update"] for row in measured) / len(measured),
                            late_harmful_update_rate=sum(row["late_harmful_update"] for row in late) / len(late) if late else None,
                            late_event_count=len(late),
                            acceptance_rate=sum(row["update_accepted"] for row in measured) / len(measured))
    temporary = directory / "metrics.tmp"
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(directory / "metrics.json")
    return record
