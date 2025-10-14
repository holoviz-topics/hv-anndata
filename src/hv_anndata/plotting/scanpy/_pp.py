from __future__ import annotations

from typing import TYPE_CHECKING

import holoviews as hv

from hv_anndata import ACCESSOR as A
from hv_anndata.plotting import utils

if TYPE_CHECKING:
    from anndata import AnnData


def highest_expr_genes(
    adata: AnnData,
    n_top: int = 20,
    *,
    layer: str | None = None,
    gene_symbols: str | None = None,
) -> hv.BoxWhisker:
    hxg = utils.highest_expr_genes(adata, n_top, layer=layer, gene_symbols=gene_symbols)
    hxg_melted = hxg.to_df().melt(var_name="gene", value_name="frac_pct")
    return hv.BoxWhisker(hxg_melted, ["gene"], ["frac_pct"]).opts(
        ylabel="% of total counts", invert_axes=True
    )


def highly_variable_genes(adata: AnnData) -> hv.Layout:
    d1, d2 = (
        ("variances", "variances_norm")
        if adata.uns["hvg"]["flavor"] == "seurat_v3"
        else ("dispersions", "dispersions_norm")
    )

    return hv.Layout([
        hv.Scatter(adata, [A.var["means"]], [A.var[d], A.var["highly_variable"]]).opts(
            color=A.var["highly_variable"],
            cmap={True: "black", False: "gray"},
            legend_labels={
                True: "highly variable",
                False: "not highly variable",
            },
            legend_position="bottom_right",
            xlabel="mean expression of genes",
            ylabel=f"{d1} of genes ({'' if 'norm' in d else 'not '}normalized)",
        )
        for d in [d2, d1]
    ])


def scrublet_score_distribution(adata: AnnData) -> hv.Layout:
    labels = dict(
        xlabel="Doublet score",
        ylabel="Probability density",
    )

    observed = (
        hv.Dataset(adata, [], [A.obs["doublet_score"]])
        .hist(A.obs["doublet_score"], adjoin=False)
        .opts(
            xlim=(0, 1),
            logy=True,
            ylim=(1, None),
            title="Observed transcriptomes",
            **labels,
        )
    )

    doublets_sim = (
        hv.Table(adata.uns["scrublet"]["doublet_scores_sim"], "scores")
        .hist("scores", adjoin=False)
        .opts(xlim=(0, 1), shared_axes=False, title="Simulated doublets", **labels)
    )

    return hv.Layout([observed, doublets_sim]) * hv.VLine(
        adata.uns["scrublet"]["threshold"]
    )
