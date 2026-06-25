"""HOD, assembly-bias, phase-space, and component-based IA-HOD namespace."""

from __future__ import annotations

from typing import Any

from ia_analysis._lazy_imports import ExportMap, load_export

_PUBLIC = (
    "HODCatalog", "HODMeasurement", "IAComponentMeasurement", "IAStrengthParameters",
    "IAComponentModel", "ComponentIAHODModel", "standardize_hod_catalog", "measure_hod",
    "measure_assembly_hod", "measure_phase_space_hod", "build_reference_bank",
    "measure_alignment_hod_components", "predict_zheng_hod", "decorated_hod_prediction",
    "predict_ia_component", "fit_ia_component_model", "measure_pairwise_ia",
)
_EXPORTS: ExportMap = {name: ("ia_analysis.hod.api", name) for name in _PUBLIC}

__all__ = [
    *_PUBLIC, "api", "catalog", "statistics", "models", "assembly", "environment",
    "concentration", "phase_space", "ia_reference", "ia_measurements", "ia_models",
    "ia_forward", "fitting", "covariance", "plotting", "io",
]


def __getattr__(name: str) -> Any:
    """Resolve the compact public HOD API lazily."""
    return load_export(_EXPORTS, name)
