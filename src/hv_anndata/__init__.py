"""Anndata interface for holoviews."""

from __future__ import annotations

from .components import AutoCompleteMultiChoice
from .interface import AnnDataInterface, register
from .manifoldmap import ManifoldMap, create_manifoldmap_plot
from .plotting import Dotmap

__all__ = [
    "AnnDataInterface",
    "AutoCompleteMultiChoice",
    "Dotmap",
    "ManifoldMap",
    "create_manifoldmap_plot",
    "register",
]
