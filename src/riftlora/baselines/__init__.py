"""Federated LoRA baselines used by the matched Week 9 runner."""

from riftlora.baselines.spectral_aggregation import (
    aggregate_spectral_factors,
    spectral_async_aggregate,
)

from riftlora.baselines.fed_lora import (
    FedExResidualController,
    fedavg_aggregate_factor_state,
    fedex_aggregate_factor_state,
)
from riftlora.baselines.florg import (
    FlorgState,
    attach_florg_adapters,
    capture_florg_state,
    florg_aggregate_state,
    load_florg_state,
    mask_florg_gradients,
)
from riftlora.baselines.flora import (
    flora_stack_aggregate_state,
    flora_stack_aggregate_states,
    flora_stack_factor_states,
)

__all__ = [
    "aggregate_spectral_factors",
    "spectral_async_aggregate",
    "FedExResidualController",
    "FlorgState",
    "attach_florg_adapters",
    "capture_florg_state",
    "fedavg_aggregate_factor_state",
    "fedex_aggregate_factor_state",
    "flora_stack_aggregate_state",
    "flora_stack_aggregate_states",
    "flora_stack_factor_states",
    "florg_aggregate_state",
    "load_florg_state",
    "mask_florg_gradients",
]
