"""Memory-bounded primitives for scaling VAST-LoRA experiments."""

from vastlora.scale.coordinator import (
    TransportConfig,
    TransportResult,
    aggregate_compact_state,
    transport_compact_update,
    zero_compact,
)
from vastlora.scale.peft_bridge import (
    FactorSnapshot,
    capture_factor_snapshot,
    compact_factor_innovations,
    empty_adapter_state,
    fedrot_aggregate_factor_state,
    load_compact_adapter_state,
    mask_inactive_rank_gradients,
    named_peft_lora_modules,
)

__all__ = [
    "TransportConfig",
    "TransportResult",
    "FactorSnapshot",
    "aggregate_compact_state",
    "capture_factor_snapshot",
    "compact_factor_innovations",
    "empty_adapter_state",
    "fedrot_aggregate_factor_state",
    "load_compact_adapter_state",
    "mask_inactive_rank_gradients",
    "named_peft_lora_modules",
    "transport_compact_update",
    "zero_compact",
]
