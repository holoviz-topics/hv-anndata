"""Anndata interface for holoviews."""

from __future__ import annotations

from .components import GeneSelector
from .interface import ACCESSOR as _A
from .interface import AnnDataInterface, register
from .manifoldmap import ManifoldMap, ManifoldMapConfig, create_manifoldmap_plot
from .plotting import Dotmap

ACCESSOR = _A
"""Accessor for anndata.

>>> from hv_anndata import ACCESSOR as A
>>> A.layers["counts"][:, "gene-3"]  # 1D access
>>> A[:, :]  # gridded
"""

__all__ = [
    "ACCESSOR",
    "AnnDataInterface",
    "Dotmap",
    "GeneSelector",
    "ManifoldMap",
    "ManifoldMapConfig",
    "create_manifoldmap_plot",
    "register",
]
