"""Anndata interface for holoviews."""

from __future__ import annotations

from .interface import AnnDataInterface, register
from .plotting import Dotmap

__all__ = ["AnnDataInterface", "Dotmap", "register"]
