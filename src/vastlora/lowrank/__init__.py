"""Low-rank algebra primitives used by the VAST-LoRA simulator."""

from vastlora.lowrank.core import (
    CompactSVD,
    LowRankMatrix,
    build_temporal_reference,
    compatibility_scores,
    compact_svd,
    exact_lora_innovation,
    project_to_reference,
    recompress,
    weighted_sum,
)

__all__ = [
    "CompactSVD",
    "LowRankMatrix",
    "build_temporal_reference",
    "compatibility_scores",
    "compact_svd",
    "exact_lora_innovation",
    "project_to_reference",
    "recompress",
    "weighted_sum",
]
