from __future__ import annotations

from typing import TYPE_CHECKING

import holoviews as hv

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
