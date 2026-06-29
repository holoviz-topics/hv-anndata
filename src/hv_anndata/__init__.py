"""Anndata interface for holoviews."""

from __future__ import annotations

from contextlib import suppress

from . import data
from ._ref import A, AdDim
from .interface import (
    AnnDataGriddedInterface,
    AnnDataInterface,
    Dims,
    register,
)

A = A  # noqa: PLW0127, RUF067
"""Accessor for anndata.

>>> from hv_anndata import A
>>> A.layers["counts"][:, "gene-3"]  # 1D access
>>> A.X[:, :]  # gridded
"""

__all__ = [
    "A",
    "AdDim",
    "AnnDataGriddedInterface",
    "AnnDataInterface",
    "Dims",
    "data",
    "register",
]

with suppress(ImportError):  # noqa: RUF067
    from .components import GeneSelector
    from .plotting import (
        ClusterMap,
        ClusterMapConfig,
        Dotmap,
        DotmapParams,
        ManifoldMap,
        ManifoldMapConfig,
        create_clustermap_plot,
        create_manifoldmap_plot,
        dotmap_from_manifoldmap,
        labeller,
        scanpy,
    )

    __all__ += [
        "ClusterMap",
        "ClusterMapConfig",
        "Dotmap",
        "DotmapParams",
        "GeneSelector",
        "ManifoldMap",
        "ManifoldMapConfig",
        "create_clustermap_plot",
        "create_manifoldmap_plot",
        "dotmap_from_manifoldmap",
        "labeller",
        "scanpy",
    ]
