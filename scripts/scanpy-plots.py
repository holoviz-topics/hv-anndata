"""List missing Scanpy plots in the gallery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import scanpy

if TYPE_CHECKING:
    from nbformat_types import Document


DEPRECATED = {"spatial", "pca_scatter"}
UNDOCUMENTED = {"timeseries", "timeseries_as_heatmap", "timeseries_subplot"}
"""Not currently documented or planned to be documented."""
NOT_PLOTTING = {"set_rcParams_defaults", "set_rcParams_scanpy"}
"""Lives in scanpy.pl, but doesn’t draw anything."""
IGNORED = DEPRECATED | UNDOCUMENTED | NOT_PLOTTING


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
    if k not in IGNORED
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
