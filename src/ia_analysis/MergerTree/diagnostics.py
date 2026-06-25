"""Tabular diagnostics for persisted merger-tree and cross-time products."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd


def product_tables(products: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """Return all DataFrame-like tables contained in a workflow product mapping."""
    output: dict[str, pd.DataFrame] = {}
    for name, value in products.items():
        if isinstance(value, pd.DataFrame):
            output[str(name)] = value.copy()
        elif isinstance(value, list) and value and all(isinstance(row, Mapping) for row in value):
            output[str(name)] = pd.DataFrame(value)
    return output


def combine_product_tables(
    product_sets: Iterable[Mapping[str, Any]],
    table_name: str,
    *,
    source_labels: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Concatenate one named table from several workflow product mappings."""
    products = list(product_sets)
    labels = list(source_labels or (f"set_{index}" for index in range(len(products))))
    if len(labels) != len(products):
        raise ValueError("`source_labels` must match the number of product sets")
    frames: list[pd.DataFrame] = []
    for label, product in zip(labels, products):
        table = product_tables(product).get(table_name)
        if table is not None:
            table.insert(0, "source", label)
            frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_numeric_table(
    table: pd.DataFrame,
    *,
    group_by: Iterable[str] = ("snap",),
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Compute count, median, mean, and standard deviation by workflow keys."""
    groups = [column for column in group_by if column in table]
    numeric = list(columns or table.select_dtypes(include=[np.number]).columns)
    numeric = [column for column in numeric if column not in groups]
    if not numeric:
        return pd.DataFrame()
    if not groups:
        summary = table[numeric].agg(["count", "median", "mean", "std"]).T.reset_index(names="quantity")
        return summary
    return table.groupby(groups, dropna=False)[numeric].agg(["count", "median", "mean", "std"]).reset_index()


def pi_closure_residuals(
    table: pd.DataFrame,
    *,
    total_column: str = "pi_total",
    component_columns: Iterable[str] = ("pi_shape", "pi_tide", "pi_flow"),
) -> pd.DataFrame:
    """Add summed-component and fractional closure residual columns."""
    components = [column for column in component_columns if column in table]
    if total_column not in table or not components:
        raise KeyError("Closure table must include the total and at least one component column")
    output = table.copy()
    output["pi_component_sum"] = output[components].sum(axis=1)
    output["pi_closure_residual"] = output[total_column] - output["pi_component_sum"]
    denominator = np.maximum(np.abs(output[total_column].to_numpy(dtype=float)), 1.0e-30)
    output["pi_closure_fractional_residual"] = output["pi_closure_residual"].to_numpy(dtype=float) / denominator
    return output


__all__ = ["product_tables", "combine_product_tables", "summarize_numeric_table", "pi_closure_residuals"]
