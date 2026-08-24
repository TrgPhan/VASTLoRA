import torch
from torch import nn

from vastlora.lora import (
    add_dense_innovation,
    get_local_innovations,
    get_server_adapter_state,
    inject_diagnostic_lora,
    local_adapter_parameters,
    reset_local_adapters,
    reset_local_adapters_from_server,
    set_server_adapter_state,
    zero_local_adapters,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(5, 4)
        self.output = nn.Linear(4, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output(torch.tanh(self.query(value)))


def test_diagnostic_lora_starts_from_exact_server_model() -> None:
    torch.manual_seed(2)
    model = TinyModel()
    inputs = torch.randn(3, 5)
    expected = model(inputs)

    assert inject_diagnostic_lora(model, target_suffixes=("query",)) == ("query",)
    reset_local_adapters(model, rank=3, alpha=3.0, seed=11)

    torch.testing.assert_close(model(inputs), expected)
    assert sum(parameter.numel() for parameter in local_adapter_parameters(model)) == 27


def test_local_innovation_can_be_applied_to_server_state() -> None:
    torch.manual_seed(3)
    model = TinyModel()
    inject_diagnostic_lora(model, target_suffixes=("query",))
    reset_local_adapters(model, rank=2, alpha=2.0, seed=19)
    with torch.no_grad():
        for parameter in local_adapter_parameters(model):
            parameter.add_(0.1)

    state = get_server_adapter_state(model)
    innovation = get_local_innovations(model)
    updated = add_dense_innovation(state, innovation, weight=0.5)
    set_server_adapter_state(model, updated)

    torch.testing.assert_close(
        get_server_adapter_state(model)["query"],
        0.5 * innovation["query"].dense(),
    )


def test_server_factor_warm_start_has_zero_initial_innovation() -> None:
    torch.manual_seed(5)
    model = TinyModel()
    inputs = torch.randn(4, 5)
    inject_diagnostic_lora(model, target_suffixes=("query",))
    server_state = {"query": torch.randn(4, 5) * 0.05}
    set_server_adapter_state(model, server_state)
    reset_local_adapters_from_server(model, rank=2, alpha=2.0, seed=23)
    expected = model(inputs)

    innovation = get_local_innovations(model)["query"]
    torch.testing.assert_close(innovation.dense(), torch.zeros(4, 5), atol=1e-7, rtol=0)
    zero_local_adapters(model)
    torch.testing.assert_close(model(inputs), expected)


def test_server_factor_warm_start_uses_exact_factor_difference() -> None:
    model = TinyModel()
    inject_diagnostic_lora(model, target_suffixes=("query",))
    set_server_adapter_state(model, {"query": torch.randn(4, 5) * 0.1})
    reset_local_adapters_from_server(model, rank=3, alpha=3.0, seed=29)
    module = dict(model.named_modules())["query"]
    initial = module.initial_lora_b @ module.initial_lora_a
    with torch.no_grad():
        module.lora_a.add_(0.02)
        module.lora_b.sub_(0.03)

    innovation = get_local_innovations(model)["query"].dense()
    expected = module.lora_b @ module.lora_a - initial
    torch.testing.assert_close(innovation, expected)
