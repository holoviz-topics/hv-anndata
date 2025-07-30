"""Manifold module tests."""

from __future__ import annotations

from unittest.mock import Mock, patch

import anndata as ad
import colorcet as cc
import holoviews as hv
import numpy as np
import pandas as pd
import panel_material_ui as pmui
import pytest

from hv_anndata.manifoldmap import ManifoldMap, create_manifoldmap_plot, labeller


@pytest.fixture
def sadata() -> ad.AnnData:
    n_obs = 10
    n_vars = 5
    n_dims = 2

    rng = np.random.default_rng()

    x = rng.random((n_obs, n_vars))
    obsm = {
        "X_pca": rng.random((n_obs, n_dims)),
        "X_umap": rng.random((n_obs, n_dims)),
    }
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
    return ad.AnnData(X=x, obs=obs, obsm=obsm, var=var)


@pytest.mark.usefixtures("bokeh_backend")
@pytest.mark.parametrize("color_kind", ["categorical", "continuous"])
def test_create_manifoldmap_plot_no_datashading(
    sadata: ad.AnnData, color_kind: str
) -> None:
    if color_kind == "categorical":
        color_var = "cell_type"
    elif color_kind == "continuous":
        color_var = "expression_level"
    plot = create_manifoldmap_plot(
        sadata.obsm["X_umap"],
        sadata.obs[color_var].values,
        0,
        1,
        color_var,
        "UMAP1",
        "UMAP2",
        datashading=False,
    )
    assert plot.kdims == ["UMAP1", "UMAP2"]
    assert plot.vdims == [color_var]
    plot_opts = plot.opts.get("plot").kwargs
    style_opts = plot.opts.get("style").kwargs
    assert style_opts["color"] == color_var
    assert style_opts["size"] == 1
    assert style_opts["alpha"] == 0.5
    assert plot_opts["padding"] == 0
    assert len(plot_opts["tools"]) == 3
    assert "hover" in plot_opts["tools"]
    assert plot_opts["legend_position"] == "right"
    assert plot_opts["min_width"] == 300
    assert plot_opts["min_height"] == 300
    assert plot_opts["responsive"]

    if color_kind == "categorical":
        assert (
            style_opts["cmap"] == cc.b_glasbey_category10[:2] == ["#1f77b3", "#ff7e0e"]
        )
        assert plot_opts["show_legend"] is True
        assert plot_opts["colorbar"] is False
    elif color_kind == "continuous":
        assert style_opts["cmap"] == "viridis"
        assert plot_opts["show_legend"] is False
        assert plot_opts["colorbar"] is True


@pytest.mark.usefixtures("bokeh_backend")
@pytest.mark.parametrize("color_kind", ["categorical", "continuous"])
def test_create_manifoldmap_plot_datashading(
    sadata: ad.AnnData, color_kind: str
) -> None:
    if color_kind == "categorical":
        color_var = "cell_type"
    elif color_kind == "continuous":
        color_var = "expression_level"
    plot = create_manifoldmap_plot(
        sadata.obsm["X_umap"],
        sadata.obs[color_var].values,
        0,
        1,
        color_var,
        "UMAP1",
        "UMAP2",
        datashading=True,
    )

    if color_kind == "categorical":
        legend = plot.callback.inputs[0].callback.inputs[0].callback.inputs[1]
        assert legend.keys() == ["A", "B"]
        assert all(legend[color].label == color for color in ["A", "B"])
        assert (
            legend["A"].opts.get("style").kwargs["color"] == cc.b_glasbey_category10[0]
        )
        assert (
            legend["B"].opts.get("style").kwargs["color"] == cc.b_glasbey_category10[1]
        )

        dm = plot.callback.inputs[0].callback.inputs[0].callback.inputs[0]
        rop = dm.callback.inputs[0].callback.inputs[0].callback.operation
        assert rop.name == "rasterize"
        assert rop.p.aggregator.cat_column == color_var
        assert rop.p.selector.column == "UMAP1"
        dop = dm.callback.inputs[0].callback.operation
        assert dop.name == "dynspread"
        assert dop.p.threshold == 0.5
    elif color_kind == "continuous":
        dm = plot.callback.inputs[0].callback.inputs[0].callback.inputs[0]
        rop = dm.callback.inputs[0].callback.operation
        assert rop.name == "rasterize"
        assert rop.p.aggregator.__class__.__name__ == "mean"
        assert rop.p.aggregator.column == color_var
        dop = dm.callback.operation
        assert dop.name == "dynspread"
        assert dop.p.threshold == 0.5


@pytest.mark.usefixtures("bokeh_backend")
def test_manifoldmap_initialization_default(sadata: ad.AnnData) -> None:
    mm = ManifoldMap(adata=sadata)

    assert mm.param.reduction.objects == ["X_pca", "X_umap"]
    assert mm.color_by_dim == "obs"
    assert mm.reduction == "X_pca"
    assert mm.color_by == "cell_type"
    assert mm._color_options == {
        "obs": ["cell_type", "expression_level"],
        "cols": ["gene_0", "gene_1", "gene_2", "gene_3", "gene_4"],
    }


@pytest.mark.usefixtures("bokeh_backend")
def test_manifoldmap_initialization_color_by(sadata: ad.AnnData) -> None:
    mm = ManifoldMap(adata=sadata, color_by_dim="cols", color_by="gene_1")

    assert mm.param.reduction.objects == ["X_pca", "X_umap"]
    assert mm.color_by_dim == "cols"
    assert mm.reduction == "X_pca"
    assert mm.color_by == "gene_1"
    assert mm._color_options == {
        "obs": ["cell_type", "expression_level"],
        "cols": ["gene_0", "gene_1", "gene_2", "gene_3", "gene_4"],
    }


@pytest.mark.usefixtures("bokeh_backend")
def test_manifoldmap_get_dim_labels(sadata: ad.AnnData) -> None:
    mm = ManifoldMap(adata=sadata)

    assert mm.get_dim_labels("X_umap") == ["UMAP1", "UMAP2"]
    assert mm.get_dim_labels("X_pca") == ["PCA1", "PCA2"]


@pytest.mark.usefixtures("bokeh_backend")
@patch("hv_anndata.manifoldmap.create_manifoldmap_plot")
def test_manifoldmap_create_plot(mock_cmp: Mock, sadata: ad.AnnData) -> None:
    mm = ManifoldMap(adata=sadata)

    mm.create_plot(
        dr_key="X_pca",
        x_value="PCA1",
        y_value="PCA2",
        color_by_dim="obs",
        color_by="cell_type",
        datashade_value=False,
        show_labels=True,
        cmap=["#1f77b3", "#ff7e0e"],
    )
    mock_cmp.assert_called_once_with(
        sadata.obsm["X_pca"],
        sadata.obs["cell_type"].values,
        0,
        1,
        "cell_type",
        "PCA1",
        "PCA2",
        categorical=True,
        width=300,
        height=300,
        datashading=False,
        show_labels=True,
        title="PCA.cell_type",
        cmap=["#1f77b3", "#ff7e0e"],
        responsive=True,
    )


@pytest.mark.usefixtures("bokeh_backend")
def test_manifoldmap_panel_layout(sadata: ad.AnnData) -> None:
    mm = ManifoldMap(adata=sadata)

    layout = mm.__panel__()

    assert isinstance(layout, pmui.layout.Row)
    assert len(layout) == 2


def test_labeller() -> None:
    df = pd.DataFrame(
        {
            "UMAP1": [0, 1, 2, 3, 10],
            "UMAP2": [0, 1, 2, 3, 10],
            "cell_type": ["a", "a", "b", "b", "b"],
        }
    )
    dataset = hv.Dataset(df, kdims=["UMAP1", "UMAP2"], vdims=("cell_type"))
    ldm = labeller(dataset, min_count=0)
    labels = ldm[()]
    expected_data = pd.DataFrame(
        {
            "cell_type": ["b", "a"],
            "count": [3, 2],
            "x": [5, 0.5],
            "y": [5, 0.5],
        }
    )
    pd.testing.assert_frame_equal(
        labels.data.sort_values("cell_type"),
        expected_data.sort_values("cell_type"),
    )
