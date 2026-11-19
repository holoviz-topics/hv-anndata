"""Configuration file for the Sphinx documentation builder."""

from __future__ import annotations

import os
import sys
from importlib.metadata import metadata
from pathlib import Path
from subprocess import run

from scverse_doc.registry import intersphinx

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "ext"))
# inherited by notebook kernels, so `sc.datasets.*` don’t download into their CWD
os.environ["SCANPY_DATASETDIR"] = str(HERE / "_build" / "data")
_info = metadata("hv-anndata")

# specify project details
master_doc = "index"
project = _info.get("Name")

source_repository = "https://github.com/holoviz-topics/hv-anndata"

# basic build settings
extensions = [
    "scverse_doc",
    "sphinx_autodoc_typehints",
    "sphinx_exec_jupyter",
    "sphinx_issues",
    "paramdoc",
    "has_attr_test",
    "napoleon_attrdoc_fix",
]
templates_path = ["_templates"]
# not a scverse project: no scverse footer, ecosystem dropdown or announcement
html_theme_options = dict(
    article_header_end=[
        "theme-switcher",
        "navbar-icon-links",
        "toggle-secondary-sidebar",
    ],
    footer_start=["copyright", "last-updated"],
    announcement="",
)
suppress_warnings = [
    "mystnb.unknown_mime_type",
]

intersphinx_mapping = intersphinx("scanpy") | dict(
    holoviews=("https://holoviews.org/", None),
    panel=("https://panel.holoviz.org/", None),
)

always_use_bars_union = True
typehints_defaults = "comma"

# myst_nb settings
nb_execution_mode = "cache"
nb_execution_show_tb = True
nb_execution_timeout = 60  # seconds

holoviews_backends = ["bokeh", "matplotlib", "plotly"]
exec_jupyter_code = """
import hv_anndata, matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
# https://github.com/tqdm/tqdm/issues/1927
from functools import partialmethod
import tqdm.notebook
tqdm.notebook.tqdm.__init__ = partialmethod(tqdm.notebook.tqdm.__init__, disable=True)
"""
exec_jupyter_kernel = "hv-anndata"

run(["hatch", "-v", "run", "docs:install-kernel"], check=True)

# autodoc/autosummary
autodoc_default_options = {
    "members": False,
}
