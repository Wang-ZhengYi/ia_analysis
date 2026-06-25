"""Concentration-specific fixed-mass HOD helpers."""

from __future__ import annotations

from typing import Any

from ia_analysis.hod.assembly import measure_assembly_hod, split_by_secondary_property_quantiles


def split_by_concentration_quantiles(mass: Any, concentration: Any, **kwargs: Any):
    return split_by_secondary_property_quantiles(mass, concentration, **kwargs)


def measure_concentration_hod(halos: Any, galaxies: Any | None = None, **kwargs: Any):
    return measure_assembly_hod(halos, galaxies, secondary_property="concentration", **kwargs)


__all__ = ["split_by_concentration_quantiles", "measure_concentration_hod"]
