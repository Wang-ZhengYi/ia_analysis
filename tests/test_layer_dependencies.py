"""Enforce the scientific package dependency direction."""

from __future__ import annotations

import ast
from pathlib import Path


LAYER = {
    "catalogs": 0,
    "shapes": 1,
    "tides": 1,
    "dynamics": 2,
    "MergerTree": 2,
    "orbits": 2,
    "spectra": 3,
    "correlations": 3,
    "covariance": 3,
    "meshes": 3,
    "visualization": 4,
}


def _domain(module_name: str) -> str | None:
    parts = module_name.split(".")
    if len(parts) >= 2 and parts[0] == "ia_analysis":
        return parts[1]
    return None


def test_domain_imports_follow_layer_direction():
    root = Path(__file__).resolve().parents[1] / "src" / "ia_analysis"
    violations: list[str] = []
    for path in root.rglob("*.py"):
        source_domain = path.relative_to(root).parts[0]
        if source_domain not in LAYER:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend((node.lineno, alias.name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append((node.lineno, node.module))
        for line, module_name in imports:
            target_domain = _domain(module_name)
            if target_domain in LAYER and LAYER[target_domain] > LAYER[source_domain]:
                violations.append(
                    f"{path.relative_to(root)}:{line}: {source_domain} imports downstream {target_domain}"
                )
    assert not violations, "\n".join(violations)
