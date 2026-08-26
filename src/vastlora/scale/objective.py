from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from vastlora.lowrank import CompactSVD
from vastlora.scale.peft_bridge import named_peft_lora_modules


@dataclass(frozen=True)
class ComponentScoreResult:
    """First-order loss scores for compact LoRA update components."""

    scores: dict[str, torch.Tensor]
    calibration_loss: float

    @property
    def total_rank(self) -> int:
        return sum(value.numel() for value in self.scores.values())

    @property
    def retained_rank(self) -> int:
        return sum(int((value > 0.0).sum().item()) for value in self.scores.values())

    @property
    def positive_fraction(self) -> float:
        total = self.total_rank
        return self.retained_rank / total if total else 0.0

    @property
    def predicted_gain(self) -> float:
        return float(
            sum(value[value > 0.0].sum().item() for value in self.scores.values())
        )


def score_compact_components_with_hooks(
    model: torch.nn.Module,
    innovations: Mapping[str, CompactSVD],
    batch: Mapping[str, torch.Tensor],
) -> ComponentScoreResult:
    """Score rank-one update components without materializing dense gradients.

    For a compact component sigma_j u_j v_j^T, we attach a temporary forward
    perturbation with scalar coefficient c_j. The gradient dL/dc_j at c_j = 0
    equals sigma_j u_j^T G v_j, so RIFT's predicted gain is -dL/dc_j.
    """

    modules = named_peft_lora_modules(model)
    if set(modules) != set(innovations):
        missing = sorted(set(modules) - set(innovations))
        extra = sorted(set(innovations) - set(modules))
        raise ValueError(f"innovation/module mismatch; missing={missing}, extra={extra}")

    coefficients: dict[str, torch.Tensor] = {}
    handles = []
    for name, module in modules.items():
        compact = innovations[name]
        if compact.rank == 0:
            coefficients[name] = torch.empty(0, dtype=torch.float32)
            continue

        device = _module_device(module)
        u = compact.u.detach().to(device=device, dtype=torch.float32)
        s = compact.s.detach().to(device=device, dtype=torch.float32)
        v = compact.v.detach().to(device=device, dtype=torch.float32)
        coefficient = torch.zeros(compact.rank, device=device, dtype=torch.float32)
        coefficient.requires_grad_(True)
        coefficients[name] = coefficient
        handles.append(
            module.register_forward_hook(
                _make_component_hook(u=u, s=s, v=v, coefficient=coefficient)
            )
        )

    try:
        model.zero_grad(set_to_none=True)
        model.eval()
        loss = model(**batch).loss
        active = [value for value in coefficients.values() if value.requires_grad]
        grads = torch.autograd.grad(loss, active, allow_unused=True) if active else []
        grad_by_id = {
            id(coefficient): grad
            for coefficient, grad in zip(active, grads)
            if grad is not None
        }
        scores: dict[str, torch.Tensor] = {}
        for name, coefficient in coefficients.items():
            if coefficient.requires_grad:
                grad = grad_by_id.get(id(coefficient))
                if grad is None:
                    scores[name] = torch.zeros_like(coefficient, device="cpu")
                else:
                    scores[name] = (-grad.detach()).to(device="cpu", dtype=torch.float32)
            else:
                scores[name] = coefficient.detach().clone()
        return ComponentScoreResult(
            scores=scores,
            calibration_loss=float(loss.detach().float().item()),
        )
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)


def filter_compact_by_scores(
    innovations: Mapping[str, CompactSVD],
    scores: Mapping[str, torch.Tensor],
    *,
    minimum_predicted_gain: float = 0.0,
    keep_nonpositive: bool = False,
) -> dict[str, CompactSVD]:
    if set(innovations) != set(scores):
        raise ValueError("innovations and scores must have the same keys")
    if minimum_predicted_gain < 0.0:
        raise ValueError("minimum_predicted_gain must be non-negative")

    filtered: dict[str, CompactSVD] = {}
    for name, compact in innovations.items():
        layer_scores = scores[name].to(device=compact.s.device)
        if layer_scores.numel() != compact.rank:
            raise ValueError(f"score rank mismatch for {name}")
        keep = layer_scores > minimum_predicted_gain
        if keep_nonpositive:
            keep = torch.ones_like(keep, dtype=torch.bool)
        filtered[name] = CompactSVD(compact.u[:, keep], compact.s[keep], compact.v[:, keep])
    return filtered


def scale_compact_update(update: CompactSVD, scale: float) -> CompactSVD:
    if scale == 0.0:
        return CompactSVD(update.u[:, :0], update.s[:0], update.v[:, :0])
    return CompactSVD(update.u, update.s * scale, update.v)


def _make_component_hook(
    *,
    u: torch.Tensor,
    s: torch.Tensor,
    v: torch.Tensor,
    coefficient: torch.Tensor,
):
    def hook(_module, inputs, output):
        hidden = inputs[0].to(dtype=torch.float32)
        projected = torch.matmul(hidden, v)
        addition = torch.matmul(projected * (s * coefficient), u.T)
        return output + addition.to(dtype=output.dtype)

    return hook


def _module_device(module: torch.nn.Module) -> torch.device:
    for parameter in module.parameters(recurse=False):
        return parameter.device
    for parameter in module.parameters():
        return parameter.device
    return torch.device("cpu")
