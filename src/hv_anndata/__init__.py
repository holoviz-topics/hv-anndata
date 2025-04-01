"""Anndata interface for holoviews."""

from __future__ import annotations

from .featuremap import FeatureMapApp, create_featuremap_plot
from .interface import AnnDataInterface, register
from .plotting import Dotmap

__all__ = [
    "AnnDataInterface",
    "Dotmap",
    "FeatureMapApp",
    "create_featuremap_plot",
    "register",
]
