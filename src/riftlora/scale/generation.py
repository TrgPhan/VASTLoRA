"""Bounded instruction QA data and response-only generative evaluation for Week 9."""
from __future__ import annotations

import hashlib
import math

import torch
from torch.nn import functional as F


def prompt_text(item):
    context = item.get("context", "").strip()
    return (f"### Instruction:\n{item['instruction'].strip()}\n"
            + (f"### Context:\n{context}\n" if context else "")
            + "### Response:\n")


def normalized(text):
    return " ".join(str(text).split()).casefold()


def prompt_ids(tokenizer, item, config):
    if config.get("prompt_format", "plain_v1") == "plain_v1":
        return tokenizer(prompt_text(item), add_special_tokens=True)["input_ids"]
    if config["prompt_format"] != "chat_v1" or not getattr(tokenizer, "chat_template", None):
        raise ValueError("chat_v1 requires the pinned tokenizer's chat template")
    content = item["instruction"].strip()
    context = item.get("context", "").strip()
    if context:
        content += "\n\nContext:\n" + context
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
    )


def stop_token_ids(model, tokenizer):
    configured = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    ids = configured if isinstance(configured, (list, tuple)) else [configured]
    return sorted({int(i) for i in [*ids, tokenizer.eos_token_id] if i is not None})


def group_split(item, salt):
    # Context-sharing questions stay together even when instructions differ.
    key = normalized(item.get("context", "")) or normalized(item["instruction"])
    digest = hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()
    bucket = int(digest[:8], 16) / 2**32
    return ("train" if bucket < 0.7 else "validation" if bucket < 0.85 else "test"), digest


def encode_example(tokenizer, item, config, max_length):
    prompt = prompt_ids(tokenizer, item, config)
    target = tokenizer(item["response"].strip(), add_special_tokens=False)["input_ids"]
    if not prompt or not target or tokenizer.eos_token_id is None:
        raise ValueError("generation needs nonempty prompt/response and an EOS token")
    target = target + [tokenizer.eos_token_id]
    if (len(prompt) > config["max_prompt_tokens"] or len(target) > config["max_response_tokens"]
            or len(prompt) + len(target) > max_length):
        raise ValueError("generation example exceeds locked token budget; do not silently truncate")
    return prompt, target


def prepare_records(records, tokenizer, config, max_length):
    if tokenizer.eos_token_id is None:
        raise ValueError("generation requires an EOS token")
    # Fail on a broken tokenizer/protocol, rather than miscounting every row as overlength.
    prompt_ids(tokenizer, {"instruction": "check", "context": ""}, config)
    categories = config["categories"]
    splits = {name: [] for name in ("train", "validation", "test")}
    seen = set()
    audit = {"source_rows": len(records), "duplicate_prompts": 0, "empty_rows": 0,
             "overlength_rows": 0, "excluded_category": 0, "split_salt": config["split_salt"],
             "prompt_format": config.get("prompt_format", "plain_v1")}
    for index, row in enumerate(records):
        if row["category"] not in categories:
            audit["excluded_category"] += 1
            continue
        if not row["instruction"].strip() or not row["response"].strip():
            audit["empty_rows"] += 1
            continue
        identity = normalized(row["instruction"]) + "\n" + normalized(row.get("context", ""))
        if identity in seen:
            audit["duplicate_prompts"] += 1
            continue
        seen.add(identity)
        try:
            encode_example(tokenizer, row, config, max_length)
        except ValueError:
            audit["overlength_rows"] += 1
            continue
        split, group = group_split(row, config["split_salt"])
        splits[split].append({**row, "label": categories.index(row["category"]),
                              "source_id": index, "group_id": group})
    audit["split_sizes"] = {k: len(v) for k, v in splits.items()}
    audit["split_id_sha256"] = {
        k: hashlib.sha256(",".join(str(x["source_id"]) for x in v).encode()).hexdigest()
        for k, v in splits.items()
    }
    audit["definition"] = "length-filtered short QA; group split by context, or instruction when context is absent"
    return splits, audit


def load_instruction_data(config):
    from datasets import Dataset, DatasetDict, load_dataset
    from transformers import AutoTokenizer
    ds, model = config["dataset"], config["model"]
    raw = load_dataset(ds["hub_path"], revision=ds["revision"], split="train")
    tokenizer = AutoTokenizer.from_pretrained(model["name"], revision=model.get("revision"))
    splits, audit = prepare_records(raw, tokenizer, ds, model["max_length"])
    if any(not rows for rows in splits.values()):
        raise ValueError(f"empty generation split: {audit}")
    return DatasetDict({k: Dataset.from_list(v) for k, v in splits.items()}), audit


def reserve_grouped_splits(train, *, reserve_fn, split_sizes, max_train_examples, seed, **kwargs):
    """Reserve calibration roles without sharing exact contexts with another role."""
    remaining, calibration = train, []
    for index, size in enumerate(split_sizes):
        if size == 0:
            calibration.append(None)
            continue
        if len(remaining) < size:
            raise ValueError("not enough context-disjoint calibration data")
        _, selected = reserve_fn(remaining, split_sizes=(size,), max_train_examples=0,
                                 seed=seed + index, **kwargs)
        subset = selected[0]
        calibration.append(subset)
        used = set(subset["group_id"])
        remaining = remaining.select([i for i, group in enumerate(remaining["group_id"]) if group not in used])
    if len(remaining) < max_train_examples:
        raise ValueError("not enough context-disjoint client data")
    return remaining.shuffle(seed=seed).select(range(max_train_examples)), calibration


def collate_generation(tokenizer, examples, config, max_length):
    encoded = [encode_example(tokenizer, item, config, max_length) for item, _ in examples]
    width = max(len(p) + len(t) for p, t in encoded)
    rows, masks, labels = [], [], []
    for prompt, target in encoded:
        padding = width - len(prompt) - len(target)
        rows.append(prompt + target + [tokenizer.pad_token_id] * padding)
        masks.append([1] * (len(prompt) + len(target)) + [0] * padding)
        labels.append([-100] * len(prompt) + target + [-100] * padding)
    return {"input_ids": torch.tensor(rows), "attention_mask": torch.tensor(masks),
            "labels": torch.tensor(labels)}


def response_loss_values(model, batch, suffix_logits):
    logits, labels = suffix_logits(model, batch)
    mask = labels.ne(-100)
    counts = mask.sum(-1)
    if torch.any(counts == 0):
        raise ValueError("empty response loss")
    losses = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100, reduction="none")
    return (losses * mask).sum(-1), counts


def response_mean_loss(model, batch, suffix_logits):
    sums, counts = response_loss_values(model, batch, suffix_logits)
    # Equal example weights align the differentiable calibration loss with paired gates.
    return (sums / counts).mean()


@torch.no_grad()
def evaluate_generation(model, tokenizer, dataset, *, dataset_config, max_length, batch_size, suffix_logits):
    from rouge_score.rouge_scorer import RougeScorer
    model.eval()
    device = next(model.parameters()).device
    scorer = RougeScorer(["rougeL"], use_stemmer=True)
    max_new_tokens = dataset_config.get("max_new_tokens", dataset_config["max_response_tokens"])
    stop_ids = stop_token_ids(model, tokenizer)
    if not len(dataset) or not stop_ids:
        raise ValueError("generation evaluation requires examples and stop tokens")
    rows = []
    for start in range(0, len(dataset), batch_size):
        items = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        batch = collate_generation(tokenizer, [(item, 0) for item in items], dataset_config, max_length)
        sums, counts = response_loss_values(model, {k: v.to(device) for k, v in batch.items()}, suffix_logits)
        for offset, (item, nll_sum, tokens) in enumerate(zip(items, sums, counts)):
            prompt, _ = encode_example(tokenizer, item, dataset_config, max_length)
            inputs = torch.tensor([prompt], device=device)
            generated = model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
                num_beams=1, max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=stop_ids,
                use_cache=True,
            )[0, len(prompt):]
            prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
            reference = item["response"].strip()
            rouge = scorer.score(reference, prediction)["rougeL"].fmeasure
            rows.append({"eval_index": start + offset, "source_id": item["source_id"],
                         "category": item["category"], "reference": reference, "prediction": prediction,
                         "nll_sum": float(nll_sum), "response_tokens": int(tokens),
                         "response_nll": float(nll_sum / tokens), "rouge_l": rouge,
                         "exact_match": int(normalized(reference) == normalized(prediction)),
                         "generated_tokens": len(generated),
                         "hit_generation_limit": bool(len(generated) == max_new_tokens
                                                      and generated[-1].item() not in stop_ids)})
    nll = sum(r["nll_sum"] for r in rows) / sum(r["response_tokens"] for r in rows)
    mean = lambda key: sum(r[key] for r in rows) / len(rows)
    return {
        "accuracy": None, "balanced_accuracy": None, "brier": None, "class_nll": None,
        "binary_nll": None, "label_nll": None, "eos_nll": None, "nll": nll,
        "token_nll": nll, "perplexity": math.exp(nll) if nll < 700 else None,
        "mean_example_nll": mean("response_nll"), "rouge_l": mean("rouge_l"),
        "exact_match": mean("exact_match"), "generation_limit_rate": mean("hit_generation_limit"),
        "response_tokens": sum(r["response_tokens"] for r in rows),
    }, rows
