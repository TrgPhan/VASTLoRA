"""Diagnostic collection and Week-4 kill-test helpers."""

from vastlora.diagnostics.geometry import (
    GeometryResult,
    PersistentProjectionResult,
    analyze_innovation_geometry,
    transport_innovations,
    residual_budget_transport,
    persistent_temporal_projection,
    subspace_lattice_transport,
)
from vastlora.diagnostics.analysis import (
    analyze_scope,
    decide_gate,
    matched_tau_analysis,
    partial_spearman,
)
from vastlora.diagnostics.schema import (
    REQUIRED_DIAGNOSTIC_COLUMNS,
    validate_diagnostic_dataframe,
)
from vastlora.diagnostics.objective import (
    PairedGateResult,
    RankwiseFilterResult,
    filter_rankwise_by_gradient,
    paired_loss_gate,
)
from vastlora.diagnostics.competitors import (
    ProjectedCompetitorUpdate,
    dense_state_difference,
    fedrot_aggregate_diagnostic_state,
    fedsteer_cached_vector_projection,
    glora_cached_consensus_projection,
)

__all__ = [
    "GeometryResult",
    "PersistentProjectionResult",
    "PairedGateResult",
    "RankwiseFilterResult",
    "ProjectedCompetitorUpdate",
    "REQUIRED_DIAGNOSTIC_COLUMNS",
    "analyze_scope",
    "analyze_innovation_geometry",
    "decide_gate",
    "matched_tau_analysis",
    "partial_spearman",
    "filter_rankwise_by_gradient",
    "dense_state_difference",
    "fedrot_aggregate_diagnostic_state",
    "fedsteer_cached_vector_projection",
    "glora_cached_consensus_projection",
    "paired_loss_gate",
    "persistent_temporal_projection",
    "residual_budget_transport",
    "subspace_lattice_transport",
    "transport_innovations",
    "validate_diagnostic_dataframe",
]
