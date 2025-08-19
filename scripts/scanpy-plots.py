"""List missing Scanpy plots in the gallery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import scanpy

if TYPE_CHECKING:
    from nbformat_types import Document


DEPRECATED = {"spatial"}
NOT_PLOTTING = {"ranking", "set_rcParams_defaults", "set_rcParams_scanpy"}


def get_content(nb: Document) -> str:
    """Concatenate all Markdown content of a notebook."""
    return "\n".join(
        line
        for cell in nb["cells"]
        if cell["cell_type"] == "markdown"
        for line in cell["source"]
    )


project_dir = Path(__file__).parent.parent

funcs = {
    k: fn
    for k in vars(scanpy.pl)
    if callable(fn := getattr(scanpy.pl, k))
    if not isinstance(fn, type)
    if k not in {*DEPRECATED, *NOT_PLOTTING}
}

nbs_contents = {
    p.stem: get_content(json.loads(p.read_text()))
    for p in (project_dir / "docs" / "examples" / "scanpy").glob("*.ipynb")
}

for func in funcs:
    if any(func in content for content in nbs_contents.values()):
        print(f"Found   {{func}}`scanpy.pl.{func}`")
    else:
        print(f"Missing {{func}}`scanpy.pl.{func}`")
