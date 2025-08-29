"""Clustermap module tests."""

from __future__ import annotations

from unittest.mock import Mock, patch

import anndata as ad
import numpy as np
import pandas as pd
import panel_material_ui as pmui
import pytest

from hv_anndata.clustermap import ClusterMap, create_clustermap_plot


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


@pytest.mark.usefixtures("bokeh_backend")
def test_create_clustermap_plot_invalid_obs_keys(sadata: ad.AnnData) -> None:
    """Test error handling for invalid observation keys."""
    with pytest.raises(ValueError, match="obs_keys 'invalid_key' not found"):
        create_clustermap_plot(
            sadata,
            obs_keys="invalid_key",
            use_raw=False,
        )


@pytest.mark.usefixtures("bokeh_backend")
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        pytest.param({}, {}, id="default"),
        pytest.param(
            dict(obs_keys="expression_level", use_raw=False),
            dict(obs_keys="expression_level", use_raw=False),
            id="custom_obs_keys",
        ),
    ],
)
def test_clustermap_initialization(
    sadata: ad.AnnData, kwargs: dict[str, object], expected: dict[str, object]
) -> None:
    """Test ClusterMap initialization with various parameters."""
    cm = ClusterMap(adata=sadata, **kwargs)

    assert cm.param.obs_keys.objects == expected.get(
        "obs_keys_objects", ["cell_type", "expression_level"]
    )
    assert cm.obs_keys == expected.get("obs_keys", "cell_type")
    assert cm.use_raw == expected.get(
        "use_raw", True
    )  # Should default to True since raw exists
    assert cm.max_genes == expected.get("max_genes", 50)
    assert cm.width == expected.get("width", 600)
    assert cm.height == expected.get("height", 400)


@pytest.mark.usefixtures("bokeh_backend")
def test_clustermap_update_on_obs_keys(sadata: ad.AnnData) -> None:
    """Test that changing obs_keys updates colormap appropriately."""
    cm = ClusterMap(adata=sadata)

    # Start with categorical (cell_type)
    initial_categorical = cm._categorical
    initial_cmap_options = list(cm.param.cmap.objects.keys())

    # Change to continuous
    cm.obs_keys = "expression_level"
    assert cm._categorical != initial_categorical
    assert list(cm.param.cmap.objects.keys()) != initial_cmap_options


@pytest.mark.usefixtures("bokeh_backend")
@patch("hv_anndata.clustermap.create_clustermap_plot")
def test_clustermap_create_plot(mock_ccp: Mock, sadata: ad.AnnData) -> None:
    """Test ClusterMap create_plot method calls underlying function correctly."""
    cm = ClusterMap(adata=sadata)

    cm.create_plot(
        obs_keys="cell_type",
        use_raw=False,
        cmap=["#1f77b3", "#ff7e0e"],
        max_genes=20,
    )

    mock_ccp.assert_called_once_with(
        sadata,
        obs_keys="cell_type",
        use_raw=False,
        max_genes=20,
        width=600,
        height=400,
        cmap=["#1f77b3", "#ff7e0e"],
        title="Clustered Heatmap - cell_type",
    )


@pytest.mark.usefixtures("bokeh_backend")
def test_clustermap_panel_layout(sadata: ad.AnnData) -> None:
    """Test ClusterMap Panel layout creation."""
    cm = ClusterMap(adata=sadata)

    layout = cm.__panel__()

    assert isinstance(layout, pmui.layout.Row)
    assert len(layout) == 2  # Widgets + plot view


@pytest.mark.usefixtures("bokeh_backend")
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
