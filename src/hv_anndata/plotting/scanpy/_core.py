"""Scanpy plots."""

from __future__ import annotations

import re
from functools import partial
from typing import TYPE_CHECKING, TypeVar, cast, overload

import holoviews as hv
import numpy as np
import pandas as pd
import scanpy as sc
from fast_array_utils import stats

from hv_anndata import ACCESSOR as A
from hv_anndata.accessors import AdPath, GraphVecAcc, LayerVecAcc, MultiVecAcc

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable
    from typing import Literal

    from anndata import AnnData
    from pandas.api.extensions import ExtensionArray

    # TODO: export in scanpy: https://github.com/scverse/scanpy/issues/3826
    AggType = Literal["count_nonzero", "mean", "sum", "var", "median"]


__all__ = [
    "heatmap",
    "matrixplot",
    "scatter",
    "stacked_violin",
    "tracksplot",
    "umap",
    "violin",
]


def scatter(
    adata: AnnData,
    /,
    kdims: Collection[AdPath],
    vdims: Collection[AdPath] = (),
    *,
    color: AdPath | None = None,
) -> hv.Scatter:
    """Shortcut for a scatter plot.

    Basically just

    >>> i, j = components
    >>> hv.Scatter(adata, kdims[0], [kdims[1], *vdims]).opts(aspect="square", ...)

    If ``color`` is set, it’s both added to ``vdims`` and in ``.opts(color=...)``.

    Examples
    --------

    ..  holoviews::

        import hv_anndata.plotting.scanpy as hv_sc
        from hv_anndata import data, register, ACCESSOR as A

        register()

        adata = data.pbmc68k_processed()
        hv_sc.scatter(adata, A[:, ["PSAP", "C1QA"]], color=A.obs["bulk_labels"]).opts(
            cmap="tab10", show_legend=False
        )

    Returns
    -------
    A scatter plot object

    """
    try:
        i, j = kdims
    except ValueError:
        msg = "kdims must have length 2"
        raise ValueError(msg) from None

    if color is not None:
        vdims = [*vdims, color]
    sc = hv.Scatter(adata, i, [j, *vdims])
    if color is not None:
        sc = sc.opts(color=color)

    return sc.opts(aspect="square", legend_position="right")


def _scatter(
    kdims: Collection[AdPath],
    adata: AnnData,
    /,
    vdims: Collection[AdPath] = (),
    *,
    color: AdPath | None = None,
) -> hv.Scatter:
    __tracebackhide__ = True
    return scatter(adata, kdims, vdims, color=color)


umap = partial(_scatter, A.obsm["X_umap"][:, [0, 1]])


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

    Set ``base`` to ``A`` or ``A.layers[key]``,
    and ``transpose=True`` to switch the order of the axes.

    If ``add_dendrogram`` is True, the dendrogram is added.
    Call it directly to customize the dendrogram:

    >>> hv.operation.dendrogram(heatmap, adjoint_dims=..., main_dim=base[:, :])

    Examples
    --------

    ..  holoviews::

        import hv_anndata.plotting.scanpy as hv_sc
        from hv_anndata import data, register, ACCESSOR as A

        register()

        adata = data.pbmc68k_processed()
        markers = ["C1QA", "PSAP", "CD79A", "CD79B", "CST3", "LYZ"]
        hv_sc.heatmap(
            adata[:, markers], A, [A.obs["n_counts"]], add_dendrogram="obs"
        ).opts(hv.opts.HeatMap(xticks=0, aspect=2))

    Returns
    -------
    A heatmap object

    """
    kdims = (
        [getattr(A, base.ax[:-1]).index] * 2
        if isinstance(base, GraphVecAcc)
        else [A.obs.index, A.var.index]
    )
    if transpose:
        kdims.reverse()
    hm = hv.HeatMap(adata, kdims, [base[:, :], *vdims])
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
    """Tracksplot.

    Examples
    --------

    ..  holoviews::

        import hv_anndata.plotting.scanpy as hv_sc
        from hv_anndata import data, register, ACCESSOR as A

        register()

        adata = data.pbmc68k_processed()
        markers = ["C1QA", "PSAP", "CD79A", "CD79B", "CST3", "LYZ"]
        hv_sc.tracksplot(
            adata, markers, color=A.obs["bulk_labels"]
        ).opts(hv.opts.Curve(aspect=20))

    Returns
    -------
    A :class:`~holoviews.NdLayout` containing :class:`~holoviews.Curve` objects.

    """
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

    If ``vdims`` is an ``AdPath``, a single violin is returned:

    >>> hv.Violin(adata, kdims, [vdims, color]).opts(violin_fill_color=color, ...)

    Otherwise, a layout is returned:

    >>> hv.Layout([violin(adata, kdims, vdim, ...) for vdim in vdims]).opts(...)

    Examples
    --------

    ..  holoviews::

        import hv_anndata.plotting.scanpy as hv_sc
        from hv_anndata import data, register, ACCESSOR as A

        register()

        adata = data.pbmc68k_processed()
        hv_sc.violin(adata, A.obs[["percent_mito", "n_counts", "n_genes"]]).opts(
            hv.opts.Violin(ylim=(0, None))
        )

    ..  holoviews::

        import hv_anndata.plotting.scanpy as hv_sc
        from hv_anndata import data, register, ACCESSOR as A

        register()

        adata = data.pbmc68k_processed()
        hv_sc.violin(adata, A.obs["S_score"], color=A.obs["bulk_labels"]).opts(
            width=500, xrotation=30
        )

    """
    if not isinstance(vdims, AdPath):
        vdims = list(vdims)
        if not all(isinstance(vdim, AdPath) for vdim in vdims):
            msg = f"vdims must be an AdPath or a collection of AdPaths, got {vdims!r}."
            raise TypeError(msg)
        return hv.Layout([
            violin(adata, vdim, color=color).opts(title=vdim.label, ylabel="")
            for vdim in vdims
        ]).opts(axiswise=True)

    kdims = list(kdims)
    if color and color not in kdims:
        kdims.append(color)
    opts = dict(violin_fill_color=color) if color else {}
    return hv.Violin(adata, kdims, vdims).opts(**opts, ylabel=vdims.label)


def stacked_violin(adata: AnnData, /, xdim: AdPath, ydim: AdPath) -> hv.GridSpace:
    """Stacked violin plot.

    Groups data by `xdim` and `ydim` and then plots a single violin for each group.

    Examples
    --------

    ..  holoviews::

        import hv_anndata.plotting.scanpy as hv_sc
        from hv_anndata import data, register, ACCESSOR as A

        register()

        adata = data.pbmc68k_processed()
        markers = ["C1QA", "PSAP", "CD79A", "CD79B", "CST3", "LYZ"]
        hv_sc.stacked_violin(
            adata[:, markers], A.var.index, A.obs["bulk_labels"]
        ).opts(hv.opts.Violin(aspect="square"))

    Returns
    -------
    A :class:`~holoviews.GridSpace` containing :class:`~holoviews.Violin` objects.

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
        case "var", "obs":
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
        [xdim, ydim],
    )


def matrixplot(
    adata: AnnData,
    /,
    group_by: AdPath,
    *,
    func: AggType = "mean",
    data: LayerVecAcc | MultiVecAcc = A,
    add_totals: bool = False,
) -> hv.HeatMap | hv.AdjointLayout:
    """Heatmap with totals per column.

    Examples
    --------

    ..  holoviews::

        import hv_anndata.plotting.scanpy as hv_sc
        from hv_anndata import data, register, ACCESSOR as A

        register()

        adata = data.pbmc68k_processed()
        markers = ["C1QA", "PSAP", "CD79A", "CD79B", "CST3", "LYZ"]
        hv_sc.matrixplot(
            adata[:, markers], A.obs["bulk_labels"], data=A.layers["counts"],
            add_totals=True
        )

    Returns
    -------
    A heatmap.
    If ``add_totals`` is True, a :class:`~holoviews.AdjointLayout` is returned
    containing the heatmap and a :class:`~holoviews.Bars` object.

    """
    # TODO: make AdPath inspectable: https://github.com/holoviz-topics/hv-anndata/pull/87
    if match := re.fullmatch(r"A\.(obs|var)\['(\w+)'\]", str(group_by)):
        axis, by = cast("tuple[Literal['obs', 'var'], str]", match.groups())
    else:
        msg = f"`by` needs to be `A.obs['…']` or `A.var['…']`, got {group_by!r}"
        raise TypeError(msg)

    layer = obsm = varm = None
    if isinstance(data, LayerVecAcc):
        layer = data.k
    elif not isinstance(data, MultiVecAcc):
        msg = (
            "`data` needs to be `A[:, :]` or `A.{layers,obsm,varm}['…'][:, :]`, "
            f"got {data!r}"
        )
        raise TypeError(msg)
    elif data.ax == "obsm":
        obsm = data.k
    elif data.ax == "varm":
        varm = data.k

    agg = sc.get.aggregate(
        adata, by, func, axis=axis, layer=layer, obsm=obsm, varm=varm
    )
    agg.var["totals"] = stats.sum(agg.layers[func], axis=0)
    heatmap = hv.HeatMap(
        agg,
        [A.obs.index, A.var.index],
        [A.layers[func][:, :]],
    ).opts(xrotation=30)
    if not add_totals:
        return _add_hover(heatmap)
    bars = hv.Bars(agg, A.var.index, A.var["totals"]).opts(
        yticks=0,
        xlabel="",  #  TODO: holoviews issue  # noqa: TD003
    )
    return hv.AdjointLayout([_add_hover(heatmap), _add_hover(bars)])


def _get_categories(
    vals: ExtensionArray | np.ndarray,
) -> Iterable[str | int | float]:
    if isinstance(vals, np.ndarray):
        return np.unique(vals)
    if isinstance(vals, pd.Categorical):
        return vals.categories[vals.categories.isin(vals)]
    return vals.unique()


_D = TypeVar("_D", bound=hv.core.dimension.Dimensioned)


def _add_hover(obj: _D) -> _D:
    if hv.Store.current_backend == "bokeh":
        return obj.opts(tools=["hover"])
    return obj
