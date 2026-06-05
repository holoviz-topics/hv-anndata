"""Configuration file for the Sphinx documentation builder."""

from __future__ import annotations

import sys
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
    "has_attr_test",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
nitpicky = True
suppress_warnings = [
    "mystnb.unknown_mime_type",
    "ref.class",  # auto-generated docs
]

intersphinx_mapping = dict(
    anndata=("https://anndata.readthedocs.io/en/stable/", None),
    holoviews=("https://holoviews.org/", None),
    numpy=("https://numpy.org/doc/stable/", None),
    pandas=("https://pandas.pydata.org/pandas-docs/stable/", None),
    panel=("https://panel.holoviz.org/", None),
    python=("https://docs.python.org/3/", None),
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
exec_jupyter_code = "import hv_anndata"
exec_jupyter_kernel = "hv-anndata"

run(["hatch", "-v", "run", "docs:install-kernel"], check=True)

# autodoc/autosummary
autodoc_default_options = {
    "members": False,
}
