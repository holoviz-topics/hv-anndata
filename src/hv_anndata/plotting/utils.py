"""Statistical utilities used in plotting."""

from __future__ import annotations

import numpy as np
import scanpy as sc
from anndata import AnnData
from fast_array_utils import stats


def highest_expr_genes(
    adata: AnnData,
    n_top: int = 20,
    *,
    layer: str | None = None,
    gene_symbols: str | None = None,
) -> AnnData:
    """Get top n genes by mean expression."""
    norm_expr = sc.pp.normalize_total(
        adata, target_sum=100, layer=layer, inplace=False
    )["X"]
    mean_percent = stats.mean(norm_expr, axis=0)
    top_idx = np.argsort(mean_percent)[::-1][:n_top]
    counts_top_genes = norm_expr[:, top_idx]
    var_labels = (
        adata.var_names[top_idx]
        if gene_symbols is None
        else adata.var[gene_symbols].iloc[top_idx].astype("string")
    )
    return AnnData(counts_top_genes, adata.obs_names.to_frame(), var_labels.to_frame())
