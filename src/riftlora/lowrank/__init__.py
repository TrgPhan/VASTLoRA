"""Low-rank algebra primitives used by the VAST-LoRA simulator."""

from riftlora.lowrank.core import (
    AdaptiveReference,
    CompactSVD,
    LowRankMatrix,
    build_adaptive_temporal_reference,
    build_temporal_reference,
    compatibility_scores,
    compact_svd,
    exact_lora_innovation,
    project_to_reference,
    recompress,
    weighted_sum,
)

__all__ = [
    "AdaptiveReference",
    "CompactSVD",
    "LowRankMatrix",
    "build_adaptive_temporal_reference",
    "build_temporal_reference",
    "compatibility_scores",
    "compact_svd",
    "exact_lora_innovation",
    "project_to_reference",
    "recompress",
    "weighted_sum",
]

