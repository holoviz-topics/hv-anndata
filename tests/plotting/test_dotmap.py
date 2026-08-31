"""Test plotting."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from hv_anndata import Dotmap
from hv_anndata.plotting.dotmap import _RADIUS_PER_PERCENT, _sizebar_ticks

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.usefixtures("bokeh_renderer")
@pytest.mark.parametrize(
    "marker_func", [lambda x: {"group A": x}, list], ids=["dict", "list"]
)
def test_dotmap_bokeh(marker_func: Callable) -> None:
    adata = sc.datasets.pbmc68k_reduced()
    markers = ["C1QA", "PSAP", "CD79A", "CD79B", "CST3", "LYZ"]

    dotmap_layout = Dotmap(
        adata=adata, marker_genes=marker_func(markers), groupby="bulk_labels"
    )
    dotmap = dotmap_layout.plot()

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


@pytest.mark.usefixtures("mpl_renderer")
@pytest.mark.parametrize(
    "marker_func", [lambda x: {"group A": x}, list], ids=["dict", "list"]
)
def test_dotmap_mpl(marker_func: Callable) -> None:
    adata = sc.datasets.pbmc68k_reduced()
    markers = ["C1QA", "PSAP", "CD79A", "CD79B", "CST3", "LYZ"]

    dotmap_layout = Dotmap(
        adata=adata, marker_genes=marker_func(markers), groupby="bulk_labels"
    )
    dotmap = dotmap_layout.plot()

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


@pytest.mark.usefixtures("bokeh_renderer")
def test_dotmap_use_raw_explicit_bokeh() -> None:
    """Test explicit use_raw settings with bokeh backend."""
    adata = sc.datasets.pbmc68k_reduced()
    markers = ["C1QA", "PSAP"]

    # Test use_raw=True without raw (should raise error)
    adata.raw = None
    dotmap_layout = Dotmap(
        adata=adata,
        marker_genes={"A": markers},
        groupby="bulk_labels",
        use_raw=True,
    )
    with pytest.raises(
        ValueError, match=r"use_raw=True but \.raw attribute is not present"
    ):
        dotmap_layout.plot()


@pytest.mark.usefixtures("bokeh_renderer")
def test_dotmap_all_missing_genes_bokeh() -> None:
    """Test error when all genes are missing with bokeh backend."""
    adata = sc.datasets.pbmc68k_reduced()

    dotmap_layout = Dotmap(
        adata=adata, marker_genes={"A": ["FAKE1", "FAKE2"]}, groupby="bulk_labels"
    )

    with pytest.raises(
        ValueError, match="None of the specified marker genes are present"
    ):
        dotmap_layout.plot()


@pytest.mark.usefixtures("bokeh_renderer")
def test_dotmap_duplicate_genes_bokeh() -> None:
    adata = sc.datasets.pbmc68k_reduced()
    sel_marker_genes = {"A": ["FCN1"], "B": ["FCN1"]}
    dotmap_layout = Dotmap(
        adata=adata, marker_genes=sel_marker_genes, groupby="bulk_labels"
    )
    dotmap = dotmap_layout.plot()
    assert dotmap.data.shape == (20, 6)


@pytest.mark.parametrize(
    ("percentages", "expected"),
    [
        ([1.0, 12.0, 45.0, 80.0], [20.0, 40.0, 60.0, 80.0]),
        ([5.0, 100.0], [20.0, 40.0, 60.0, 80.0, 100.0]),
        ([0.2, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]),
        ([37.0, 37.0], [37.0]),
        ([42.0], [42.0]),
        ([], []),
        ([np.nan, np.nan], []),
        ([np.nan, 10.0, 50.0], [10.0, 20.0, 30.0, 40.0, 50.0]),
    ],
    ids=[
        "typical_spread",
        "up_to_full_range",
        "sub_percent_lower_bound",
        "all_equal",
        "single_value",
        "empty",
        "all_nan",
        "some_nan",
    ],
)
def test_sizebar_ticks(percentages: list[float], expected: list[float]) -> None:
    """Ticks are the values bokeh's own AdaptiveTicker picks, in radius space.

    They have to match, because SizeBarView._paint positions the legend dots
    from its adaptive ticker regardless of the ticker we supply -- ticks that
    disagree would label the wrong dots. Expectations here are in percent for
    readability and converted to radius space to compare.
    """
    ticks = _sizebar_ticks(pd.Series(percentages, dtype=float))

    assert ticks == pytest.approx([p * _RADIUS_PER_PERCENT for p in expected])


@pytest.mark.parametrize(
    "percentages",
    [[1.0, 12.0, 45.0, 80.0], [0.2, 5.0], [5.0, 100.0], [0.0, 0.0], [0.0, 60.0]],
    ids=["typical", "sub_percent", "full_range", "all_zero", "zero_lower_bound"],
)
def test_sizebar_ticks_within_glyph_radius_range(percentages: list[float]) -> None:
    """Ticks must fall inside the bar's range, which is the glyph radius extent.

    SizeBarView._paint derives that range from the actual min/max radius and
    ``bounds`` can only narrow it, so a tick outside would be positioned off the
    end of the bar. Zero is covered because it makes the lower bound degenerate.
    """
    series = pd.Series(percentages, dtype=float)
    ticks = _sizebar_ticks(series)
    r_min = series.min() * _RADIUS_PER_PERCENT
    r_max = series.max() * _RADIUS_PER_PERCENT

    assert all(math.isfinite(tick) for tick in ticks)
    # An all-equal range is widened by an epsilon, so allow a tolerance.
    assert all(r_min - 1e-6 <= tick <= r_max + 1e-6 for tick in ticks)


@pytest.mark.usefixtures("bokeh_renderer")
def test_dotmap_sizebar_has_explicit_ticker_bokeh() -> None:
    """The sizebar needs a ticker up front or its tick labels never render."""
    adata = sc.datasets.pbmc68k_reduced()
    dotmap = Dotmap(
        adata=adata, marker_genes={"A": ["C1QA", "PSAP"]}, groupby="bulk_labels"
    ).plot()

    sizebar_opts = dotmap.opts.get().kwargs["sizebar_opts"]

    assert sizebar_opts["ticker"].ticks
