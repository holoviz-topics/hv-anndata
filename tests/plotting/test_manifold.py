"""Manifold module tests."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import anndata as ad
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import pytest
from holoviews.operation.datashader import dynspread, rasterize

from hv_anndata import ACCESSOR as A
from hv_anndata import ManifoldMap, create_manifoldmap_plot
from hv_anndata.interface import register, unregister

if TYPE_CHECKING:
    from collections.abc import Iterator
    from unittest.mock import Mock


@pytest.fixture
def sadata() -> Iterator[ad.AnnData]:
    register()
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
    yield ad.AnnData(X=x, obs=obs, obsm=obsm, var=var)
    unregister()


@pytest.mark.usefixtures("bokeh_renderer")
@pytest.mark.parametrize("color_kind", ["categorical", "continuous"])
def test_create_manifoldmap_plot_no_datashading(
    sadata: ad.AnnData, color_kind: str
) -> None:
    if color_kind == "categorical":
        color_var = "cell_type"
    elif color_kind == "continuous":
        color_var = "expression_level"
    plot = create_manifoldmap_plot(
        sadata,
        "X_umap",
        color_by=color_var,
        xaxis_label="UMAP1",
        yaxis_label="UMAP2",
        datashading=False,
    )
    assert plot.kdims == [A.obsm["X_umap"][:, 0], A.obsm["X_umap"][:, 1]]
    assert plot.vdims == [A.obs[color_var]]
    plot_opts = plot.opts.get("plot").kwargs
    style_opts = plot.opts.get("style").kwargs
    assert style_opts["color"] == A.obs[color_var]
    assert style_opts["size"] == 3
    assert style_opts["alpha"] == pytest.approx(0.5)
    assert plot_opts["padding"] == 0
    assert len(plot_opts["tools"]) == 3
    assert "hover" in plot_opts["tools"]
    assert plot_opts["legend_position"] == "right"
    assert plot_opts["min_width"] == 300
    assert plot_opts["min_height"] == 300
    assert plot_opts["responsive"]

    if color_kind == "categorical":
        assert plot_opts["show_legend"] is True
        assert plot_opts["colorbar"] is False
    elif color_kind == "continuous":
        assert style_opts["cmap"] == "viridis"
        assert plot_opts["show_legend"] is False
        assert plot_opts["colorbar"] is True


@pytest.mark.usefixtures("bokeh_renderer")
@pytest.mark.parametrize("color_kind", ["categorical", "continuous"])
def test_create_manifoldmap_plot_datashading(
    sadata: ad.AnnData, color_kind: str
) -> None:
    if color_kind == "categorical":
        color_var = "cell_type"
    elif color_kind == "continuous":
        color_var = "expression_level"
    plot = create_manifoldmap_plot(
        sadata,
        "X_umap",
        color_by=color_var,
        xaxis_label="UMAP1",
        yaxis_label="UMAP2",
        datashading=True,
    )

    el = plot[()]
    rop = el.pipeline.find(rasterize, skip_nonlinked=False)
    dop = el.pipeline.find(dynspread, skip_nonlinked=False)
    if color_kind == "categorical":
        assert rop.name.startswith("datashade")
        assert rop.aggregator.cat_column == f"A.obs['{color_var}']"
    elif color_kind == "continuous":
        assert rop.name.startswith("rasterize")
        assert rop.aggregator.__class__.__name__ == "mean"
        assert rop.aggregator.column == f"A.obs['{color_var}']"
    assert dop.name.startswith("dynspread")
    assert dop.threshold == pytest.approx(0.5)


@pytest.mark.usefixtures("bokeh_renderer")
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        pytest.param({}, {}, id="default"),
        pytest.param(
            dict(color_by_dim="cols", color_by="gene_1"),
            dict(color_by_dim="cols", color_by="gene_1"),
            id="color_by",
        ),
    ],
)
def test_manifoldmap_initialization(
    sadata: ad.AnnData, kwargs: dict[str, object], expected: dict[str, object]
) -> None:
    mm = ManifoldMap(adata=sadata, **kwargs)

    assert mm.param.reduction.objects == expected.get(
        "reduction_objects", ["X_umap", "X_pca"]
    )
    assert mm.color_by_dim == expected.get("color_by_dim", "obs")
    assert mm.reduction == expected.get("reduction", "X_umap")
    assert mm.color_by == expected.get("color_by", "cell_type")
    assert mm._color_options == expected.get(
        "_color_options",
        {
            "obs": ["cell_type", "expression_level"],
            "cols": ["gene_0", "gene_1", "gene_2", "gene_3", "gene_4"],
        },
    )


@pytest.mark.usefixtures("bokeh_renderer")
def test_manifoldmap_get_dim_labels(sadata: ad.AnnData) -> None:
    mm = ManifoldMap(adata=sadata)

    assert mm.get_dim_labels("X_umap") == ["UMAP1", "UMAP2"]
    assert mm.get_dim_labels("X_pca") == ["PCA1", "PCA2"]


@pytest.mark.usefixtures("bokeh_renderer")
@patch("hv_anndata.plotting.manifoldmap.create_manifoldmap_plot")
def test_manifoldmap_create_plot(mock_cmp: Mock, sadata: ad.AnnData) -> None:
    mm = ManifoldMap(adata=sadata)

    mm.create_plot(
        dr_key="X_pca",
        x_value="PCA1",
        y_value="PCA2",
        color_by="cell_type",
        datashade_value=False,
        show_labels=True,
        cmap=["#1f77b3", "#ff7e0e"],
    )
    mock_cmp.assert_called_once_with(
        *(sadata, "X_pca", 0, 1),
        color_by="cell_type",
        xaxis_label="PCA1",
        yaxis_label="PCA2",
        categorical=True,
        width=300,
        height=300,
        datashading=False,
        show_labels=True,
        streams=[],
        title="PCA.cell_type",
        cmap=["#1f77b3", "#ff7e0e"],
        responsive=True,
        ls=None,
        labeller_opts={},
    )


@pytest.mark.usefixtures("bokeh_renderer")
def test_manifoldmap_panel_layout(sadata: ad.AnnData) -> None:
    mm = ManifoldMap(adata=sadata)

    layout = mm.__panel__()

    assert isinstance(layout, pn.Row)
    assert len(layout) == 2


@pytest.mark.usefixtures("bokeh_renderer")
def test_manifoldmap_streams(sadata: ad.AnnData) -> None:
    bounds_xy = hv.streams.BoundsXY()
    mm = ManifoldMap(adata=sadata, streams=[bounds_xy])
    mm._plot_view.clear()
    assert bounds_xy.source is None
    mm.__panel__()
    assert bounds_xy.source is not None
