"""Environment-specific fixed-mass HOD helpers."""

from __future__ import annotations

from typing import Any

from ia_analysis.hod.assembly import measure_assembly_hod, split_by_secondary_property_quantiles


def split_by_environment_quantiles(mass: Any, environment: Any, **kwargs: Any):
    return split_by_secondary_property_quantiles(mass, environment, **kwargs)


def measure_environment_hod(halos: Any, galaxies: Any | None = None, **kwargs: Any):
    return measure_assembly_hod(halos, galaxies, secondary_property="environment", **kwargs)


__all__ = ["split_by_environment_quantiles", "measure_environment_hod"]
