"""Clustermap module tests."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import panel_material_ui as pmui
import pytest
import scanpy as sc

from hv_anndata import ClusterMap


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
def test_integration() -> None:
    adata = sc.datasets.pbmc68k_reduced()  # errors

    assert ClusterMap(adata=adata).__panel__()
