"""Scanpy plots."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import holoviews as hv

from hv_anndata import ACCESSOR as A
from hv_anndata.accessors import GraphVecAcc

if TYPE_CHECKING:
    from collections.abc import Collection
    from typing import Literal

    from anndata import AnnData

    from hv_anndata.accessors import AdPath, LayerVecAcc, MultiVecAcc


def scatter(
    adata: AnnData,
    base: MultiVecAcc | LayerVecAcc | GraphVecAcc,
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

    label = f"{base.k} " if base.k is not None else ""
    return sc.opts(
        aspect="square",
        legend_position="right",
        xlabel=f"{label}{i}",
        ylabel=f"{label}{j}",
    )


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
    adata: AnnData,
    base: LayerVecAcc | GraphVecAcc = A,
    /,
    vdims: Collection[AdPath] = (),
    *,
    transpose: bool = False,
    add_dendrogram: bool | Literal["obs", "var"] = False,
) -> hv.HeatMap:
    """Shortcut for a heatmap.

    Basically just
    >>> hv.HeatMap(adata, [A.obs.index, A.var.index], [base[:, :], *vdims]).opts(...)

    Set `base` to `A` or `A.layers[key]`,
    and `transpose=True` to switch the order of the axes.

    If `add_dendrogram` is True, the dendrogram is added.
    Call it directly to customize the dendrogram:
    >>> hv.operation.dendrogram(heatmap, adjoint_dims=..., main_dim=base[:, :])

    """
    kdims = (
        [getattr(A, base.ax[:-1]).index] * 2
        if isinstance(base, GraphVecAcc)
        else [A.obs.index, A.var.index]
    )
    if transpose:
        kdims.reverse()
    xlabel, ylabel = [next(iter(d.axes)) for d in kdims]
    hm = hv.HeatMap(adata, kdims, [base[:, :], *vdims]).opts(
        xlabel=xlabel, ylabel=ylabel
    )
    if isinstance(base, GraphVecAcc):
        hm = hm.opts(aspect="square")
    if add_dendrogram:
        dims = kdims
        if isinstance(add_dendrogram, str):
            dims = [dims[0] if add_dendrogram == "obs" else dims[1]]
        hm = hv.operation.dendrogram(hm, adjoint_dims=dims, main_dim=base[:, :])
    return hm
