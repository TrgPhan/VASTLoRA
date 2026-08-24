from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a clean SST-2 warm-start checkpoint")
    parser.add_argument("--model", default="google/bert_uncased_L-2_H-128_A-2")
    parser.add_argument("--tokenizer", default="google-bert/bert-base-uncased")
    parser.add_argument("--train-examples", type=int, default=5000)
    parser.add_argument("--task", choices=("sst2", "qnli"), default="sst2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/models/bert_tiny_sst2_clean")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)

    raw = load_dataset("nyu-mll/glue", args.task)
    if not 0 < args.train_examples < len(raw["train"]):
        raise ValueError("train-examples must leave data for federated clients")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def tokenize(batch: dict[str, list[object]]) -> dict[str, object]:
        if args.task == "sst2":
            texts = [batch["sentence"]]
        else:
            texts = [batch["question"], batch["sentence"]]
        values = tokenizer(
            *texts,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )
        values["labels"] = batch["label"]
        return values

    train = raw["train"].select(range(args.train_examples)).map(
        tokenize, batched=True, remove_columns=raw["train"].column_names
    )
    validation = raw["validation"].map(
        tokenize, batched=True, remove_columns=raw["validation"].column_names
    )
    columns = ["input_ids", "attention_mask", "token_type_ids", "labels"]
    train.set_format("torch", columns=[name for name in columns if name in train.column_names])
    validation.set_format(
        "torch", columns=[name for name in columns if name in validation.column_names]
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(validation, batch_size=args.batch_size * 2)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_accuracy = -1.0
    history: list[dict[str, float | int]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.detach().item()) * len(batch["labels"])
        validation_loss, validation_accuracy = evaluate(model, validation_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss / len(train),
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(record)
        print(json.dumps(record))
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    manifest = {
        "source_model": args.model,
        "source_tokenizer": args.tokenizer,
        "dataset": f"nyu-mll/glue/{args.task}",
        "warmstart_train_indices": [0, args.train_examples - 1],
        "federated_train_start": args.train_examples,
        "seed": args.seed,
        "best_validation_accuracy": best_accuracy,
        "history": history,
    }
    (args.output_dir / "warmstart_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            output = model(**batch)
            size = len(batch["labels"])
            total_loss += float(output.loss.item()) * size
            correct += int((output.logits.argmax(dim=-1) == batch["labels"]).sum().item())
            count += size
    return total_loss / count, correct / count


if __name__ == "__main__":
    main()
