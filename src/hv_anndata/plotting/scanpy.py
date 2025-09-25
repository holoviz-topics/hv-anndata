"""Scanpy plots."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, overload

import holoviews as hv
import numpy as np
import pandas as pd

from hv_anndata import ACCESSOR as A
from hv_anndata.accessors import AdPath, GraphVecAcc

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable
    from typing import Literal

    from anndata import AnnData
    from pandas.api.extensions import ExtensionArray

    from hv_anndata.accessors import LayerVecAcc, MultiVecAcc


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


def tracksplot(
    adata: AnnData, markers: Collection[AdPath], color: AdPath | None = None
) -> hv.NdLayout:
    """Shortcut for a tracksplot."""
    more_vdims = [] if color is None else [color]
    curves = {
        m: hv.Curve(adata, [A.obs.index], [A[:, m], *more_vdims]).opts(
            xticks=0,
            xlabel="",
            ylabel=m,
            title="",
            show_legend=False,  # TODO: switch to below impl after fixing https://github.com/holoviz/holoviews/issues/5438
            aspect=2 * len(markers),
        )
        for m in markers
    }
    if color is not None:
        curves = {m: c.groupby(color, hv.NdOverlay) for m, c in curves.items()}
    return hv.NdLayout(curves, kdims=["marker"]).cols(1)


def _tracksplot2(
    adata: AnnData, markers: Collection[str], color: AdPath | None = None
) -> hv.GridSpace:
    """Tracksplot variant. Faster but Gridspace is generally buggy.

    We can switch after <https://github.com/holoviz/holoviews/issues/5438> is fixed.
    """
    assert color is not None  # noqa: S101
    return hv.GridSpace(
        {
            (0, m): hv.Curve(adata, [A.obs.index], [A[:, m], color])
            .opts(aspect=2 * len(markers))
            .groupby(A.obs["bulk_labels"], hv.NdOverlay)
            for m in markers
        },
        kdims=["_", "marker"],
    ).opts(show_legend=True, xaxis=None)


@overload
def violin(
    adata: AnnData,
    /,
    vdims: AdPath,
    *,
    kdims: Collection[AdPath] = (),
    color: AdPath | None = None,
) -> hv.Violin: ...
@overload
def violin(
    adata: AnnData,
    /,
    vdims: Collection[AdPath],
    *,
    kdims: Collection[AdPath] = (),
    color: AdPath | None = None,
) -> hv.Layout: ...
def violin(
    adata: AnnData,
    /,
    vdims: Collection[AdPath] | AdPath,
    *,
    kdims: Collection[AdPath] = (),
    color: AdPath | None = None,
) -> hv.Violin | hv.Layout:
    """Shortcut for a violin plot.

    If `vdims` is an `AdPath`, a single violin is returned:
    >>> hv.Violin(adata, kdims, [vdims, color]).opts(violin_fill_color=color, ...)

    Otherwise, a layout is returned:
    >>> hv.Layout([violin(adata, kdims, vdim, ...) for vdim in vdims]).opts(...)

    """
    if not isinstance(vdims, AdPath):
        vdims = list(vdims)
        if not all(isinstance(vdim, AdPath) for vdim in vdims):
            msg = f"vdims must be an AdPath or a collection of AdPaths, got {vdims!r}."
            raise TypeError(msg)
        return hv.Layout([
            violin(adata, vdim, color=color).opts(title=str(vdim), ylabel="")
            for vdim in vdims
        ]).opts(axiswise=True)

    kdims = list(kdims)
    if color and color not in kdims:
        kdims.append(color)
    opts = dict(violin_fill_color=color) if color else {}
    return hv.Violin(adata, kdims, vdims).opts(**opts, ylabel=str(vdims))


def stacked_violin(adata: AnnData, xdim: AdPath, ydim: AdPath) -> hv.GridSpace:
    """Stacked violin plot.

    Groups data by `xdim` and `ydim` and then plots a single violin for each group.
    """
    if len(xdim.axes) != 1 or len(ydim.axes) != 1:
        msg = "xdim and ydim must map to the same axis."
        raise ValueError(msg)
    xvals = xdim(adata)
    yvals = ydim(adata)

    match next(iter(xdim.axes)), next(iter(ydim.axes)):
        case "obs", "obs":
            idx = lambda x, y: adata[(xvals == x) & (yvals == y), :]  # noqa: E731
        case "var", "var":
            idx = lambda x, y: adata[:, (xvals == x) & (yvals == y)]  # noqa: E731
        case "obs", "var":
            idx = lambda x, y: adata[xvals == x, yvals == y]  # noqa: E731
        case "obs", "var":
            idx = lambda x, y: adata[yvals == y, xvals == x]  # noqa: E731
        case _:
            raise AssertionError

    return hv.GridSpace(
        {
            # TODO: should Violin vdim be able to be 2D?  # noqa: TD003
            (x, y): hv.Violin(idx(x, y), vdims=[A[:, :]]).opts(inner=None)
            for x in _get_categories(xvals)
            for y in _get_categories(yvals)
        },
        ["marker", "bulk label"],
    )


def _get_categories(
    vals: ExtensionArray | np.ndarray,
) -> Iterable[str | int | float]:
    if isinstance(vals, np.ndarray):
        return np.unique(vals)
    if isinstance(vals, pd.Categorical):
        return vals.categories
    return vals.unique()
