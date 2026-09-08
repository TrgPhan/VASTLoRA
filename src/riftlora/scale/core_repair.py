"""Experimental calibration repair in the incoming innovation's two-sided span."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Callable, Mapping, Sequence

import torch

from riftlora.lowrank import CompactSVD
from riftlora.scale.peft_bridge import named_peft_lora_modules
from riftlora.scale.objective import _module_device


@dataclass(frozen=True)
class CoreRepairConfig:
    steps: int = 3
    learning_rate: float = 0.03
    radius: float = 0.5
    proximal_weight: float = 0.01
    delay_scale: float = 8.0

    def validate(self) -> None:
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("core repair steps must be a positive integer")
        for name in ("learning_rate", "radius", "proximal_weight", "delay_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (
                name in {"learning_rate", "delay_scale"} and value == 0
            ):
                raise ValueError(f"invalid core repair {name}")


@dataclass(frozen=True)
class CoreRepairResult:
    updates: dict[str, CompactSVD]
    diagnostics: dict[str, float]


def repair_compact_core(
    model: torch.nn.Module,
    innovations: Mapping[str, CompactSVD],
    scores: Mapping[str, torch.Tensor],
    weighted_batches: Sequence[tuple[Mapping[str, torch.Tensor], float]],
    *,
    loss_fn: Callable,
    server_weight: float,
    staleness: int,
    diagonal_only: bool = False,
    config: CoreRepairConfig = CoreRepairConfig(),
) -> CoreRepairResult:
    """Optimize U C V^T jointly, centered on the positive-component filter.

    C is normalized by the innovation Frobenius norm, making the per-layer
    trust radius relative to incoming update energy. Hooks include the server
    step weight. The runner must gate the actual rank-projected candidate on a
    separate split: this fitting objective alone is not a safety guarantee.
    """
    config.validate()
    if not math.isfinite(server_weight) or server_weight <= 0 or staleness < 0:
        raise ValueError("server weight must be positive and staleness non-negative")
    modules = named_peft_lora_modules(model)
    if set(modules) != set(innovations) or set(scores) != set(innovations):
        raise ValueError("core repair module/innovation/score mismatch")
    batches = list(weighted_batches)
    if not batches or any(not math.isfinite(w) or w <= 0 for _, w in batches):
        raise ValueError("core repair needs finite positive microbatch weights")
    total_weight = sum(w for _, w in batches)
    radius = config.radius / math.sqrt(1.0 + staleness / config.delay_scale)
    cores, anchors, bases = {}, {}, {}
    handles = []
    modes = [(module, module.training) for module in model.modules()]
    parameters = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    try:
        model.eval()
        for parameter, _ in parameters:
            parameter.requires_grad_(False)
        for name, compact in innovations.items():
            score = scores[name].detach().float().flatten()
            if score.numel() != compact.rank or not torch.isfinite(score).all():
                raise ValueError(f"invalid core repair scores for {name}")
            if not all(torch.isfinite(x).all() for x in (compact.u, compact.s, compact.v)):
                raise ValueError(f"nonfinite innovation for {name}")
            energy = float(compact.s.float().norm())
            if not compact.rank or energy == 0:
                continue
            device = _module_device(modules[name])
            u, v = (x.detach().to(device=device, dtype=torch.float32) for x in (compact.u, compact.v))
            diagonal = compact.s.detach().to(device=device, dtype=torch.float32) / energy
            diagonal = diagonal * (score.to(device) > 0)
            anchor = diagonal if diagonal_only else torch.diag(diagonal)
            core = anchor.clone().requires_grad_(True)
            cores[name], anchors[name], bases[name] = core, anchor, (u, v, energy)

            def hook(_module, inputs, output, u=u, v=v, core=core, energy=energy):
                projected = inputs[0].float() @ v
                mixed = projected * core if diagonal_only else projected @ core.T
                return output + (server_weight * energy * (mixed @ u.T)).to(output.dtype)

            handles.append(modules[name].register_forward_hook(hook))

        active = list(cores.values())
        fit_losses = []
        if active:
            optimizer = torch.optim.Adam(active, lr=config.learning_rate)
            for _ in range(config.steps):
                optimizer.zero_grad(set_to_none=True)
                fit_loss = 0.0
                for batch, weight in batches:
                    loss = loss_fn(model, batch)
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite core repair loss")
                    fit_loss += float(loss.detach()) * weight / total_weight
                    grads = torch.autograd.grad(loss * weight / total_weight, active, allow_unused=True)
                    for core, grad in zip(active, grads):
                        if grad is not None:
                            if not torch.isfinite(grad).all():
                                raise ValueError("nonfinite core repair gradient")
                            core.grad = grad.detach() if core.grad is None else core.grad + grad.detach()
                for name, core in cores.items():
                    penalty_grad = 2 * config.proximal_weight * (core.detach() - anchors[name]) / len(active)
                    core.grad = penalty_grad if core.grad is None else core.grad + penalty_grad
                optimizer.step()
                with torch.no_grad():
                    for name, core in cores.items():
                        delta = core - anchors[name]
                        core.copy_(anchors[name] + delta * min(1.0, radius / max(float(delta.norm()), 1e-12)))
                fit_losses.append(fit_loss)

        updates = {}
        distances, off_diagonal = [], []
        for name, compact in innovations.items():
            if name not in cores:
                updates[name] = compact
                continue
            core = cores[name].detach()
            distances.append(float((core - anchors[name]).norm()))
            matrix = torch.diag(core) if diagonal_only else core
            off_diagonal.append(float((matrix - torch.diag(torch.diag(matrix))).norm()))
            left, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
            keep = singular > 0
            u, v, energy = bases[name]
            updates[name] = CompactSVD(
                (u @ left[:, keep]).cpu(), (singular[keep] * energy).cpu(),
                (v @ vh[keep].T).cpu(),
            )
        return CoreRepairResult(updates, {
            "core_radius": radius,
            "core_max_distance": max(distances, default=0.0),
            "core_mean_offdiagonal_norm": sum(off_diagonal) / max(len(off_diagonal), 1),
            "core_fit_loss_initial": fit_losses[0] if fit_losses else 0.0,
            "core_fit_loss_last_pre_step": fit_losses[-1] if fit_losses else 0.0,
            "core_fit_steps": float(config.steps if active else 0),
        })
    finally:
        for handle in handles:
            handle.remove()
        for parameter, requires_grad in parameters:
            parameter.requires_grad_(requires_grad)
        for module, training in modes:
            module.training = training
