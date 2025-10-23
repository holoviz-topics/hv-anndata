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
    """Get ``n_top`` genes by mean expression.

    Uses :func:`hv_anndata.plotting.utils.highest_expr_genes` internally.

    Examples
    --------

    ..  holoviews::

        import hv_anndata.plotting.scanpy as hv_sc
        from hv_anndata import data, register, ACCESSOR as A

        register()

        adata = data.pbmc68k_processed()
        hv_sc.highest_expr_genes(adata, layer="counts")

    Returns
    -------
    A box-and-whisker plot.

    """
    hxg = utils.highest_expr_genes(adata, n_top, layer=layer, gene_symbols=gene_symbols)
    hxg_melted = hxg.to_df().melt(var_name="gene", value_name="frac_pct")
    return hv.BoxWhisker(hxg_melted, ["gene"], ["frac_pct"]).opts(
        ylabel="% of total counts", invert_axes=True
    )


def highly_variable_genes(adata: AnnData) -> hv.Layout:
    """Plot dispersions used to identify highly variable genes.

    Examples
    --------

    ..  holoviews::

        import scanpy as sc
        import hv_anndata.plotting.scanpy as hv_sc
        from hv_anndata import data, register, ACCESSOR as A

        register()

        adata = data.pbmc68k_processed()
        sc.pp.highly_variable_genes(adata)

        hv_sc.highly_variable_genes(adata)

    Returns
    -------
    A layout containing two :class:`~holoviews.Scatter` plots,
    one normalized and one not.

    """
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
    """Plot the doublet score distribution.

    Plots the doublet score probability densities for observed transcriptomes
    and simulated doublets.

    Examples
    --------

    ..  holoviews::

        import scanpy as sc
        import hv_anndata.plotting.scanpy as hv_sc
        from hv_anndata import data, register, ACCESSOR as A

        register()

        adata = data.pbmc68k_processed()
        adata_sim = sc.pp.scrublet_simulate_doublets(adata)
        sc.pp.scrublet(adata, adata_sim)

        hv_sc.scrublet_score_distribution(adata)

    Returns
    -------
    Layout containing two histograms.

    """
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
