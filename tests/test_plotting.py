"""Test plotting."""

from __future__ import annotations

import pandas as pd
import pytest
import scanpy as sc

from hv_anndata.plotting import Dotmap


@pytest.mark.usefixtures("bokeh_backend")
def test_dotmap_bokeh() -> None:
    adata = sc.datasets.pbmc68k_reduced()
    markers = ["C1QA", "PSAP", "CD79A", "CD79B", "CST3", "LYZ"]

    dotmap = Dotmap(
        adata=adata, marker_genes={"group A": markers}, groupby="bulk_labels"
    )

    assert isinstance(dotmap.data, pd.DataFrame)
    assert dotmap.data.shape == (60, 7)
    assert list(dotmap.data.columns) == [
        "cluster",
        "percentage",
        "mean_expression",
        "marker_cluster_name",
        "gene_id",
        "marker_line",
        "mean_expression_norm",
    ]
    assert sorted(dotmap.data.gene_id.unique()) == sorted(markers)
    assert "size" in dotmap.opts.get().kwargs


@pytest.mark.usefixtures("mpl_backend")
def test_dotmap_mpl() -> None:
    adata = sc.datasets.pbmc68k_reduced()
    markers = ["C1QA", "PSAP", "CD79A", "CD79B", "CST3", "LYZ"]

    dotmap = Dotmap(
        adata=adata, marker_genes={"group A": markers}, groupby="bulk_labels"
    )

    assert isinstance(dotmap.data, pd.DataFrame)
    assert dotmap.data.shape == (60, 7)
    assert list(dotmap.data.columns) == [
        "cluster",
        "percentage",
        "mean_expression",
        "marker_cluster_name",
        "gene_id",
        "marker_line",
        "mean_expression_norm",
    ]
    assert sorted(dotmap.data.gene_id.unique()) == sorted(markers)
    assert "s" in dotmap.opts.get().kwargs
