"""Anndata interface for holoviews."""

from __future__ import annotations

from .interface import AnnDataInterface, register
from .plotting import Dotmap
from .featuremap import FeatureMapApp, create_featuremap_plot

__all__ = ["AnnDataInterface", "Dotmap", "register", "FeatureMapApp", "create_featuremap_plot"]
