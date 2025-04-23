"""Anndata interface for holoviews."""

from __future__ import annotations

from .manifoldmap import ManifoldMap, create_manifoldmap_plot
from .interface import AnnDataInterface, register
from .plotting import Dotmap

__all__ = [
    "AnnDataInterface",
    "Dotmap",
    "ManifoldMap",
    "create_manifoldmap_plot",
    "register",
]
