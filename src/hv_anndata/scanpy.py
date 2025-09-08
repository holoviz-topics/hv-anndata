"""Scanpy plots."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import holoviews as hv

from .interface import ACCESSOR as A

if TYPE_CHECKING:
    from collections.abc import Collection

    from anndata import AnnData

    from .accessors import AdPath, LayerVecAcc, MultiVecAcc


def scatter(
    adata: AnnData,
    base: MultiVecAcc | LayerVecAcc,
    /,
    components: Collection[int] | Collection[str] = (0, 1),
    vdims: Collection[AdPath] = (),
    *,
    color: AdPath | None = None,
) -> hv.Scatter:
    """Shortcut for a scatter plot.

    Basically just
    >>> i, j = components
    >>> hv.Scatter(adata, base[:, i], [base[:, j], *vdims]).opts(aspect="square", ...)

    Set `base` to `A.obsm[key]`, `A.varm[key]`, `A`, or `A.layers[key]`.

    If `color` is set, it’s both added to `vdims` and in `.opts(color=...)`.
    """
    try:
        i, j = components
    except ValueError:
        msg = "components must have length 2"
        raise ValueError(msg) from None

    if color is not None:
        vdims = [*vdims, color]
    # TODO: allow plotting obs against each other: base["o1", :]  # noqa: TD003
    sc = hv.Scatter(adata, base[:, i], [base[:, j], *vdims])
    if color is not None:
        sc = sc.opts(color=color)
    return sc.opts(aspect="square", legend_position="right")


def _scatter(
    base: MultiVecAcc,
    adata: AnnData,
    /,
    vdims: Collection[AdPath] = (),
    *,
    components: Collection[int] = (0, 1),
    color: AdPath | None = None,
) -> hv.Scatter:
    __tracebackhide__ = True
    return scatter(adata, base, components, vdims, color=color)


umap = partial(_scatter, A.obsm["X_umap"])


def heatmap(
    adata: AnnData, /, vdims: Collection[AdPath], *, transpose: bool = False
) -> hv.HeatMap:
    kdims = [A.obs.index, A.var.index]
    if transpose:
        kdims.reverse()
    return hv.HeatMap(adata, kdims, [A[:, :], *vdims]).opts(xticks=0)
