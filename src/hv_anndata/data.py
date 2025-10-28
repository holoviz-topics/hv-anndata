"""Helper functions to load example datasets from :mod:`scanpy.datasets`."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import scanpy as sc

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = ["pbmc68k_processed"]


def pbmc68k_processed() -> AnnData:
    """Load PBMC 68k reduced with some changes.

    - ``.layers["counts"]`` instead of ``raw``.
    - UMAP computed.
    """
    return _pbmc68k_processed().copy()


@cache
def _pbmc68k_processed() -> AnnData:
    adata = sc.datasets.pbmc68k_reduced()
    adata.layers["counts"] = adata.raw.X
    del adata.raw
    sc.tl.umap(adata)
    return adata
