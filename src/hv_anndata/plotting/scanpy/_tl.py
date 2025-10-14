from __future__ import annotations

from typing import TYPE_CHECKING, cast

import holoviews as hv
import numpy as np
import pandas as pd

from hv_anndata.accessors import GraphVecAcc, MultiVecAcc
from hv_anndata.interface import ACCESSOR as A

if TYPE_CHECKING:
    from typing import Literal

    from anndata import AnnData
    from scipy.sparse import csr_matrix

    from hv_anndata.accessors import AdPath


def draw_graph(
    adata: AnnData,
    kdims: list[AdPath] | MultiVecAcc,
    edge_vdim: Literal["distances", "connectivities"] | GraphVecAcc = "connectivities",
    node_vdims: AdPath | list[AdPath] | None = None,
    *,
    neighbors_key: str = "neighbors",
) -> hv.Graph:
    """Draw a graph.

    If `edge_vdim` is `"distances"`/`"connectivities"`, the graph data is retrieved like
    :func:`scanpy.pp.neighbors` stores it: `A.uns[neighbors_key][f"{edge_vdim}_key"]`.
    Therefore `.opts(edge_color=calculated_edge_vdim)` is set by default.
    """
    adata = adata.copy()
    adata.obs["cell index"] = range(adata.n_obs)
    if isinstance(kdims, MultiVecAcc):
        kdims = [kdims[0], kdims[1]]
    if isinstance(edge_vdim, str):
        edge_vdim = A.obsp[adata.uns[neighbors_key][f"{edge_vdim}_key"]]
    elif not isinstance(edge_vdim, GraphVecAcc):
        msg = f"edge_vdim must be a string or `A.obsp[key]`, got {edge_vdim!r}."
        raise TypeError(msg)

    edges = cast("csr_matrix", getattr(adata, edge_vdim.ax)[edge_vdim.k]).tocoo()
    nodes = hv.Nodes(adata, [*kdims, A.obs["cell index"]], node_vdims)
    return hv.Graph(((*edges.coords, edges.data), nodes), vdims=edge_vdim[:, :]).opts(
        edge_color=edge_vdim[:, :]
    )


def ranking(
    adata: AnnData, dim: AdPath, n_points: int = 10, *, include_lowest: bool = True
) -> hv.Labels:
    [ax] = dim.axes
    # full arrays
    scores = dim(adata)
    labels = getattr(adata, ax).index

    # subset
    idx = np.argsort(scores)
    idx_top, idx_bot = idx[-n_points:][::-1], idx[:n_points][::-1]
    scores = np.r_[scores[idx_top], np.nan, scores[idx_bot]]
    labels = np.r_[labels[idx_top], ["⋯"], labels[idx_bot]]

    # prepare
    data = pd.DataFrame(
        dict(
            rank=np.arange(n_points * 2 + 1),
            score=np.where(np.isnan(scores), np.nanmean(scores), scores),
            dot=np.r_[[True] * n_points, False, [True] * n_points].astype(float),
            text=labels,
            align=np.r_[["right"] * n_points, ["center"], ["left"] * n_points],
        )
    )
    if not include_lowest:
        i = data[:n_points]["score"].diff().abs().argmax()
        data = data[:n_points].assign(
            align=np.r_[["right"] * i, ["left"] * (n_points - i)]
        )

    return hv.Labels(data, ["rank", "score"], ["text", "align"]).opts(
        angle=90, text_align="align", xticks=0
    ) * hv.Points(data, ["rank", "score"], ["text", "dot"]).opts(alpha="dot")
