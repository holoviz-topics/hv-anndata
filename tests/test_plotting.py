"""Test plotting."""

from __future__ import annotations

import pandas as pd
import pytest
import scanpy as sc

from hv_anndata.plotting import Dotmap

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.usefixtures("bokeh_backend")
@pytest.mark.parametrize(
    "marker_func", [lambda x: {"group A": x}, list], ids=["dict", "list"]
)
def test_dotmap_bokeh(marker_func: Callable) -> None:
    adata = sc.datasets.pbmc68k_reduced()
    markers = ["C1QA", "PSAP", "CD79A", "CD79B", "CST3", "LYZ"]

    dotmap = Dotmap(
        adata=adata, marker_genes=marker_func(markers), groupby="bulk_labels"
    )

    assert isinstance(dotmap.data, pd.DataFrame)
    assert dotmap.data.shape == (60, 6)
    assert sorted(dotmap.data.columns) == [
        "cluster",
        "gene_id",
        "marker_cluster_name",
        "marker_line",
        "mean_expression",
        "percentage",
    ]
    assert sorted(dotmap.data.gene_id.unique()) == sorted(markers)
    assert "size" in dotmap.opts.get().kwargs


@pytest.mark.usefixtures("mpl_backend")
@pytest.mark.parametrize(
    "marker_func", [lambda x: {"group A": x}, list], ids=["dict", "list"]
)
def test_dotmap_mpl(marker_func: Callable) -> None:
    adata = sc.datasets.pbmc68k_reduced()
    markers = ["C1QA", "PSAP", "CD79A", "CD79B", "CST3", "LYZ"]

    dotmap = Dotmap(
        adata=adata, marker_genes=marker_func(markers), groupby="bulk_labels"
    )

    assert isinstance(dotmap.data, pd.DataFrame)
    assert dotmap.data.shape == (60, 6)
    assert sorted(dotmap.data.columns) == [
        "cluster",
        "gene_id",
        "marker_cluster_name",
        "marker_line",
        "mean_expression",
        "percentage",
    ]
    assert sorted(dotmap.data.gene_id.unique()) == sorted(markers)
    assert "s" in dotmap.opts.get().kwargs


@pytest.mark.usefixtures("bokeh_backend")
def test_dotmap_use_raw_explicit_bokeh() -> None:
    """Test explicit use_raw settings with bokeh backend."""
    adata = sc.datasets.pbmc68k_reduced()
    markers = ["C1QA", "PSAP"]

    # Test use_raw=True without raw (should raise error)
    adata.raw = None
    with pytest.raises(
        ValueError, match="use_raw=True but .raw attribute is not present"
    ):
        Dotmap(
            adata=adata,
            marker_genes={"A": markers},
            groupby="bulk_labels",
            use_raw=True,
        )


@pytest.mark.usefixtures("bokeh_backend")
def test_dotmap_all_missing_genes_bokeh() -> None:
    """Test error when all genes are missing with bokeh backend."""
    adata = sc.datasets.pbmc68k_reduced()

    with pytest.raises(
        ValueError, match="None of the specified marker genes are present"
    ):
        Dotmap(
            adata=adata, marker_genes={"A": ["FAKE1", "FAKE2"]}, groupby="bulk_labels"
        )


@pytest.mark.usefixtures("bokeh_backend")
def test_dotmap_duplicate_genes_bokeh() -> None:
    adata = sc.datasets.pbmc68k_reduced()
    sel_marker_genes = {"A": ["FCN1"], "B": ["FCN1"]}
    dotmap = Dotmap(adata=adata, marker_genes=sel_marker_genes, groupby="bulk_labels")
    assert dotmap.data.shape == (20, 6)
