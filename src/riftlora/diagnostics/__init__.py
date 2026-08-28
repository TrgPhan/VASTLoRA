"""Diagnostic collection and Week-4 kill-test helpers."""

from riftlora.diagnostics.geometry import (
    GeometryResult,
    PersistentProjectionResult,
    analyze_innovation_geometry,
    transport_innovations,
    residual_budget_transport,
    persistent_temporal_projection,
    subspace_lattice_transport,
)
from riftlora.diagnostics.analysis import (
    analyze_scope,
    decide_gate,
    matched_tau_analysis,
    partial_spearman,
)
from riftlora.diagnostics.schema import (
    REQUIRED_DIAGNOSTIC_COLUMNS,
    validate_diagnostic_dataframe,
)
from riftlora.diagnostics.objective import (
    PairedGateResult,
    RankwiseFilterResult,
    filter_rankwise_by_gradient,
    paired_loss_gate,
)
from riftlora.diagnostics.competitors import (
    COMPETITOR_SPECS,
    CompetitorSpec,
    ProjectedCompetitorUpdate,
    competitor_description,
    competitor_fidelity,
    dense_state_difference,
    fedrot_aggregate_diagnostic_state,
    fedsteer_cached_vector_projection,
    fedex_exact_diagnostic_state,
    glora_cached_consensus_projection,
)

__all__ = [
    "GeometryResult",
    "PersistentProjectionResult",
    "PairedGateResult",
    "RankwiseFilterResult",
    "CompetitorSpec",
    "COMPETITOR_SPECS",
    "ProjectedCompetitorUpdate",
    "REQUIRED_DIAGNOSTIC_COLUMNS",
    "analyze_scope",
    "analyze_innovation_geometry",
    "decide_gate",
    "matched_tau_analysis",
    "partial_spearman",
    "competitor_description",
    "competitor_fidelity",
    "filter_rankwise_by_gradient",
    "fedex_exact_diagnostic_state",
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

