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

__all__ = [
    "GeometryResult",
    "PersistentProjectionResult",
    "REQUIRED_DIAGNOSTIC_COLUMNS",
    "analyze_scope",
    "analyze_innovation_geometry",
    "decide_gate",
    "matched_tau_analysis",
    "partial_spearman",
    "persistent_temporal_projection",
    "residual_budget_transport",
    "subspace_lattice_transport",
    "transport_innovations",
    "validate_diagnostic_dataframe",
]
