"""Clustermap module tests."""

from __future__ import annotations

import asyncio

import anndata as ad
import numpy as np
import pandas as pd
import panel as pn
import panel_material_ui as pmui
import pytest
import scanpy as sc

from hv_anndata import ClusterMap, create_clustermap_plot


@pytest.fixture
def sadata() -> ad.AnnData:
    n_obs = 10
    n_vars = 5

    rng = np.random.default_rng()

    x = rng.random((n_obs, n_vars))
    obs = pd.DataFrame(
        {
            "cell_type": ["A", "B"] * (n_obs // 2),
            "expression_level": rng.random((n_obs,)),
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        index=[f"gene_{i}" for i in range(n_vars)],
    )
    # Create raw data for testing use_raw functionality
    raw_x = rng.random((n_obs, n_vars))
    raw_var = pd.DataFrame(index=[f"raw_gene_{i}" for i in range(n_vars)])
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.raw = ad.AnnData(X=raw_x, var=raw_var, obs=obs)
    return adata


@pytest.mark.usefixtures("bokeh_renderer")
def test_clustermap_panel_layout(sadata: ad.AnnData) -> None:
    """Test ClusterMap Panel layout creation."""
    cm = ClusterMap(adata=sadata)

    layout = cm.__panel__()

    assert isinstance(layout, pmui.layout.Row)
    assert len(layout) == 2  # Widgets + plot view


@pytest.mark.usefixtures("bokeh_renderer")
def test_clustermap_no_raw_data() -> None:
    """Test ClusterMap behavior when no raw data is available."""
    n_obs = 5
    n_vars = 3
    rng = np.random.default_rng()

    x = rng.random((n_obs, n_vars))
    obs = pd.DataFrame(
        {"cell_type": ["A", "B", "A", "B", "A"]},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    adata = ad.AnnData(X=x, obs=obs, var=var)  # No raw data

    cm = ClusterMap(adata=adata)
    assert cm.use_raw is False  # Should default to False when no raw data


@pytest.mark.usefixtures("bokeh_renderer")
def test_clustermap_builds_plot_synchronously_on_construction(
    sadata: ad.AnnData,
) -> None:
    """Construction alone, with no async machinery involved, must render the plot.

    Building the first plot via the async _update_plot watcher instead would
    depend on a running event loop to schedule it -- not guaranteed outside a
    live Panel session (e.g. plain scripts and tests) -- and was observed to
    leak an event loop/socket when construction happened outside of one;
    see #163.
    """
    cm = ClusterMap(adata=sadata)

    assert isinstance(cm._plot_placeholder.object, pn.pane.HoloViews)


@pytest.mark.usefixtures("bokeh_renderer")
def test_clustermap_explicit_use_raw_builds_plot_synchronously(
    sadata: ad.AnnData,
) -> None:
    """Passing use_raw explicitly must not require a later trigger to render."""
    cm = ClusterMap(adata=sadata, use_raw=True)

    assert cm.use_raw is True
    assert isinstance(cm._plot_placeholder.object, pn.pane.HoloViews)


@pytest.mark.usefixtures("bokeh_renderer")
def test_integration() -> None:
    adata = sc.datasets.pbmc68k_reduced()  # errors

    assert ClusterMap(adata=adata).__panel__()


@pytest.fixture
def sadata_mismatched_raw() -> ad.AnnData:
    """AnnData whose raw layer has far more genes than the primary var.

    Used to exercise the max_genes filtering path with use_raw=True, where
    gene-variance indices are computed over adata.raw.X's (larger) gene axis.

    Returns
    -------
    AnnData with 3 main-var genes and 20 raw-var genes.

    """
    n_obs = 10
    n_vars = 3
    n_raw_vars = 20

    rng = np.random.default_rng()

    x = rng.random((n_obs, n_vars))
    obs = pd.DataFrame(
        {"cell_type": ["A", "B"] * (n_obs // 2)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    raw_x = rng.random((n_obs, n_raw_vars))
    raw_var = pd.DataFrame(index=[f"raw_gene_{i}" for i in range(n_raw_vars)])
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.raw = ad.AnnData(X=raw_x, var=raw_var, obs=obs)
    return adata


@pytest.mark.usefixtures("bokeh_renderer")
def test_create_clustermap_plot_use_raw_var_names_from_raw(
    sadata_mismatched_raw: ad.AnnData,
) -> None:
    """max_genes filtering with use_raw must index raw.var_names, not var_names.

    Pre-fix this raised an IndexError: the filter indices are computed from
    adata.raw.X's (larger) gene axis but were used to index the (shorter)
    adata.var_names; see #163.
    """
    plot = create_clustermap_plot(sadata_mismatched_raw, use_raw=True, max_genes=10)

    genes = set(plot.dimension_values("variable", expanded=False))
    assert len(genes) == 10
    assert all(gene.startswith("raw_gene_") for gene in genes)


@pytest.mark.usefixtures("bokeh_renderer")
def test_create_clustermap_plot_heatmap_is_responsive(
    sadata_mismatched_raw: ad.AnnData,
) -> None:
    """The HeatMap must be responsive so it doesn't overflow at large gene counts."""
    plot = create_clustermap_plot(sadata_mismatched_raw, use_raw=True, max_genes=10)

    assert plot.main.opts.get("plot").kwargs["responsive"] is True


@pytest.mark.usefixtures("bokeh_renderer")
# asyncio.run() spins up and tears down a fresh loop; Panel/Bokeh's IOLoop and
# thread-pool teardown from asyncio.to_thread emit ResourceWarnings on gc that
# land here (or on whatever test runs next), unrelated to this test's outcome.
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_clustermap_update_plot_use_raw_with_mismatched_var_names(
    sadata_mismatched_raw: ad.AnnData,
) -> None:
    """The ClusterMap widget must render, not crash, through the same code path."""
    cm = ClusterMap(adata=sadata_mismatched_raw, use_raw=True, max_genes=10)

    asyncio.run(cm._update_plot())

    assert isinstance(cm._plot_placeholder.object, pn.pane.HoloViews)
    assert cm._plot_placeholder.loading is False


@pytest.mark.usefixtures("bokeh_renderer")
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_clustermap_update_plot_error_sets_alert(
    sadata: ad.AnnData, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Errors during recompute must surface as an Alert, not propagate.

    The recompute runs off the main thread via asyncio.to_thread, so an
    unhandled exception there would otherwise be silently lost; see #163.
    """

    def _boom(*_args: object, **_kwargs: object) -> None:
        msg = "kaboom"
        raise ValueError(msg)

    cm = ClusterMap(adata=sadata)
    monkeypatch.setattr("hv_anndata.plotting.clustermap.create_clustermap_plot", _boom)

    asyncio.run(cm._update_plot())

    alert = cm._plot_placeholder.object
    assert isinstance(alert, pn.pane.Alert)
    assert alert.alert_type == "danger"
    assert "Could not render clustermap" in alert.object
    assert cm._plot_placeholder.loading is False
