# Fed-LoRA baseline implementation for Week 9

## Scope

The 1.5B comparison profile is available as:

```powershell
python scripts/run_week9_generation.py --phase development `
  --profile qwen15b-fedlora --output-root outputs/week9_1_5b_fedlora
```

Use `--dry-run` first. Confirmation should use a clean frozen checkout and the
same profile with `--phase confirmation` only after the development budget is
fixed.

```powershell
python scripts/run_week9_generation.py --phase confirmation `
  --profile qwen15b-fedlora --output-root outputs/week9_1_5b_fedlora_confirmation
```

## Implemented methods

- `fedavg_lora`: standard LoRA factor-space FedAvg. The server averages padded
  `B` and `A` factors, then recompresses to the shared rank budget.
- `fedex_lora`: FedEx-LoRA's exact residual correction. It computes
  `mean(B A) - mean(B) mean(A)` and keeps the correction in a separate frozen
  residual forward path, so quantized base weights are not overwritten.
- `flora_lora`: FLoRA stacking from the official implementation. Weighted
  client products are represented by concatenating `B` horizontally and `A`
  vertically, so heterogeneous ranks are valid and no LoRA gauge is averaged.
  The current immediate-async adapter stacks the exact arriving innovation with
  the server state and recompresses only at `server_max_rank`.
- `florg`: single-matrix FLoRG backend. Each target layer trains one latent
  matrix `A`; the server averages `A.T @ A` and applies Procrustes alignment
  after eigendecomposition.
- `ffa_lora`: FFA-LoRA-style one-factor control. `A` is frozen and the
  remaining factor follows exact factor aggregation under the shared trace.
- `fedrot`: existing geometry baseline, retained as an optional comparison.
- `raw` and `freshness`: matched immediate-async controls.

## Fairness rules

All eight methods use the same Dolly data manifest, partition mode, client ranks,
compute times, asynchronous arrival trace, local steps, held-out examples and
seed. `buffer_size=1` is intentional for this cohort: it isolates the
aggregation method from the separate buffered-async protocol.

FLoRG is a different adapter parameterization, so its report must include
trainable parameter count and communication bytes. It should not be described
as ordinary two-factor LoRA. FedEx residual memory and wall-clock overhead
must also be reported.

The primary table should include held-out token NLL, ROUGE-L precision/recall/
F1, exact match, harmful update rate, late harmful update rate, acceptance or
coverage, runtime, and peak memory. Do not declare a winner from ROUGE-L alone.

## Fidelity boundary

The official FLoRA repository is
`https://github.com/ATP-1010/FederatedLLM` and the paper is
`https://arxiv.org/abs/2409.05976`. Its algorithm uses a round-level stack over
all selected clients. This repository exposes that exact stack primitive in
`flora_stack_factor_states`. The Week 9 protocol is an
immediate asynchronous trace (`buffer_size=1`), so `flora_lora` is a matched
one-arrival adaptation rather than a claim of reproducing a synchronous FLoRA
round. A grouped/buffered FLoRA confirmation should be a separate experiment,
with the same client cohort and an explicitly reported `buffer_size > 1`.
