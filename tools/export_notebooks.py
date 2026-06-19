"""Export raw notebook cells into reviewable Python scripts.

This is a project-management helper. It preserves code and a compact markdown
summary from each notebook, while commenting IPython magics and shell commands
that are not valid Python.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_NOTEBOOK_DIR = ROOT / "notebooks" / "raw_20260618"
EXPORT_DIR = ROOT / "src" / "ia_analysis" / "notebook_pipelines" / "exports"
INDEX_PATH = ROOT / "docs" / "notebook_exports.md"


def slug(name: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_").lower()
    if not value or value[0].isdigit():
        value = "nb_" + value
    return value


def source_text(source) -> str:
    if isinstance(source, list):
        return "".join(source)
    return source or ""


def sanitize_code(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        prefix = line[: len(line) - len(stripped)]
        if stripped.startswith(("%%", "%", "!")) or stripped in {"?", "??"}:
            lines.append(prefix + "# IPython-only: " + stripped)
        else:
            lines.append(line)
    return "\n".join(lines).rstrip() + ("\n" if lines else "")


def export_one(nb_path: Path) -> tuple[str, str | None, int, int, str]:
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - diagnostic path
        return nb_path.name, None, 0, 0, f"FAILED_READ: {exc}"

    py_name = slug(nb_path.stem) + "_nb.py"
    py_path = EXPORT_DIR / py_name
    code_cells = 0
    markdown_cells = 0
    parts = [
        f'"""Exported code from notebooks/raw_20260618/{nb_path.name}.\n\n'
        "This file is generated for project management and refactoring. "
        "Review before running end to end.\n"
        '"""\n\n'
    ]

    for i, cell in enumerate(nb.get("cells", []), start=1):
        ctype = cell.get("cell_type", "")
        text = source_text(cell.get("source", ""))
        if ctype == "markdown":
            markdown_cells += 1
            summary = " ".join(text.strip().split())[:500]
            if summary:
                parts.append(f"\n# %% [markdown] cell {i}\n")
                parts.append("# " + summary + "\n")
        elif ctype == "code":
            code_cells += 1
            parts.append(f"\n# %% code cell {i}\n")
            parts.append(sanitize_code(text))

    py_path.write_text("".join(parts), encoding="utf-8")
    return nb_path.name, py_name, code_cells, markdown_cells, "OK"


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    (EXPORT_DIR / "__init__.py").write_text(
        '"""Exported Python scripts generated from raw notebooks."""\n',
        encoding="utf-8",
    )

    rows = [export_one(path) for path in sorted(RAW_NOTEBOOK_DIR.glob("*.ipynb"))]
    md = [
        "# Notebook Code Export Index\n\n",
        "Generated from `notebooks/raw_20260618`. These scripts preserve notebook code cells for review and refactoring.\n\n",
        "| Notebook | Export | Code cells | Markdown cells | Status |\n",
        "|---|---|---:|---:|---|\n",
    ]
    for nb, py, code, mark, status in rows:
        md.append(f"| `{nb}` | `{py or ''}` | {code} | {mark} | {status} |\n")
    INDEX_PATH.write_text("".join(md), encoding="utf-8")
    print(f"exported {sum(1 for row in rows if row[-1] == 'OK')} notebooks to {EXPORT_DIR}")


if __name__ == "__main__":
    main()

