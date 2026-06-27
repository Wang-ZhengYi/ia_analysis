"""Structural checks for the complete HOD and IA-HOD workflow notebook."""

import ast
import json
from pathlib import Path


def test_full_hod_workflow_notebook_is_documented_and_syntactically_valid():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "pipelines" / "09_full_hod_ia_workflow.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    markdown = [cell for cell in notebook["cells"] if cell["cell_type"] == "markdown"]
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(markdown) >= 15
    assert len(code) >= 15
    assert notebook["metadata"]["kernelspec"]["name"] == "py312"
    combined_markdown = "\n".join("".join(cell["source"]) for cell in markdown)
    for section in (
        "Ordinary HOD",
        "Assembly",
        "Satellite phase-space",
        "IA reference-vector bank",
        "Component-level IA-HOD",
        "Conditional IA-strength",
        "Pairwise xi, omega, and eta",
        "Covariance",
    ):
        assert section.lower() in combined_markdown.lower()
    for index, cell in enumerate(code):
        ast.parse("".join(cell["source"]), filename=f"{path.name}:cell-{index}")
        assert cell.get("outputs", []) == []
        assert cell.get("execution_count") is None
