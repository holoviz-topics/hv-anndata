"""Anndata interface for holoviews."""

from __future__ import annotations

from .interface import AnnDataInterface, register
from .manifoldmap import ManifoldMap, create_manifoldmap_plot
from .plotting import Dotmap

__all__ = [
    "AnnDataInterface",
    "Dotmap",
    "ManifoldMap",
    "create_manifoldmap_plot",
    "register",
]
