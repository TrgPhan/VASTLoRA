"""PEFT binding for the explicitly parameterized AlignFed reference variant."""
from __future__ import annotations

import torch
from torch import nn

from riftlora.scale.peft_bridge import named_peft_lora_modules


def factor_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    parameters = {}
    for name, module in named_peft_lora_modules(model).items():
        parameters[f"{name}.lora_A.default.weight"] = module.lora_A["default"].weight
        parameters[f"{name}.lora_B.default.weight"] = module.lora_B["default"].weight
    return parameters


def snapshot_factors(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().float().cpu().clone() for k, v in factor_parameters(model).items()}


@torch.no_grad()
def load_factors(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    parameters = factor_parameters(model)
    if set(state) != set(parameters) or any(state[k].shape != p.shape for k, p in parameters.items()):
        raise ValueError("factor snapshot does not match model")
    for k, parameter in parameters.items():
        parameter.copy_(state[k].to(parameter))


class PenultimateFeatures(nn.Module):
    """Masked prompt pooling after the penultimate Qwen/Llama decoder block.

    This token pooling convention is a reproduction choice. No LM vocabulary
    logits or all-layer hidden-state stack are allocated for feature alignment.
    """
    def __init__(self, adapter: nn.Module):
        super().__init__()
        self.adapter = adapter
        transformer = adapter.get_base_model().model
        if len(transformer.layers) < 2:
            raise ValueError("penultimate features require at least two decoder layers")

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        transformer = self.adapter.get_base_model().model
        captured = []

        def capture(_module, _inputs, output):
            captured.append(output[0] if isinstance(output, tuple) else output)

        handle = transformer.layers[-2].register_forward_hook(capture)
        try:
            transformer(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        use_cache=False, output_hidden_states=False, return_dict=True)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("penultimate decoder hook did not execute")
        hidden = captured[-1].float()
        mask = batch["attention_mask"].bool()
        if "labels" in batch:
            mask = mask & batch["labels"].eq(-100)
        if torch.any(mask.sum(-1) == 0):
            raise ValueError("semantic features require at least one unmasked prompt token")
        return (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)


def features_at_factors(extractor, factors, batch):
    parameters = factor_parameters(extractor.adapter)
    if set(factors) != set(parameters):
        raise ValueError("functional factor keys must match the adapter")
    replacements = {f"adapter.{k}": value.to(parameters[k]) for k, value in factors.items()}
    return torch.func.functional_call(extractor, replacements, (batch,), strict=False)
