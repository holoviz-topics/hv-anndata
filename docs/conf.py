"""Configuration file for the Sphinx documentation builder."""

from __future__ import annotations

import sys
from functools import partial
from importlib.metadata import metadata
from pathlib import Path
from subprocess import run

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "ext"))

_info = metadata("hv-anndata")

# specify project details
master_doc = "index"
project = _info.get("Name")

# basic build settings
html_theme = "furo"
extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_exec_jupyter",
    "sphinx_design",
    "sphinx_issues",
    "scanpydoc.definition_list_typed_field",
    "scanpydoc.elegant_typehints",  # for qualname_overrides
    "myst_nb",
    "paramdoc",
]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
nitpicky = True
suppress_warnings = ["mystnb.unknown_mime_type"]

intersphinx_mapping = dict(
    python=("https://docs.python.org/3/", None),
    holoviews=("https://holoviews.org/", None),
    panel=("https://panel.holoviz.org/", None),
    anndata=("https://anndata.readthedocs.io/en/stable/", None),
    scanpy=("https://scanpy.readthedocs.io/en/latest/", None),
)

always_use_bars_union = True
typehints_defaults = "comma"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True

# myst_nb settings
nb_execution_mode = "cache"
nb_execution_show_tb = True
nb_execution_timeout = 30  # seconds

holoviews_backends = ["bokeh", "matplotlib", "plotly"]

# autodoc/autosummary
autodoc_default_options = {
    "members": False,
}


# https://github.com/executablebooks/MyST-NB/issues/574
def _patch_myst_nb() -> None:
    from jupyter_cache.executors import (  # type: ignore[import-not-found]  # noqa: PLC0415
        utils,
    )
    from myst_nb.core.execute import (  # type: ignore[import-not-found]  # noqa: PLC0415
        cache,
        direct,
    )

    run(
        ["hatch", "-v", "run", "docs:install-kernel"],
        check=True,
    )

    cache.single_nb_execution = direct.single_nb_execution = partial(
        utils.single_nb_execution, kernel_name="hv-anndata"
    )


_patch_myst_nb()
