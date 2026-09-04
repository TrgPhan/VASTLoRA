"""Memory-bounded primitives for scaling VAST-LoRA experiments."""

from riftlora.scale.coordinator import (
    TransportConfig,
    TransportResult,
    aggregate_compact_state,
    transport_compact_update,
    zero_compact,
)
from riftlora.scale.objective import (
    ComponentScoreResult,
    filter_compact_by_scores,
    scale_compact_update,
    score_compact_components_microbatched,
    score_compact_components_with_hooks,
)
from riftlora.scale.peft_bridge import (
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
    "ComponentScoreResult",
    "FactorSnapshot",
    "aggregate_compact_state",
    "capture_factor_snapshot",
    "compact_factor_innovations",
    "empty_adapter_state",
    "fedrot_aggregate_factor_state",
    "filter_compact_by_scores",
    "load_compact_adapter_state",
    "mask_inactive_rank_gradients",
    "named_peft_lora_modules",
    "scale_compact_update",
    "score_compact_components_microbatched",
    "score_compact_components_with_hooks",
    "transport_compact_update",
    "zero_compact",
]

