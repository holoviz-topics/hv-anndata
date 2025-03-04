"""Test anndata interface."""

from __future__ import annotations

import contextlib
from string import ascii_lowercase
from typing import TYPE_CHECKING

import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

from hv_anndata.interface import AnnDataInterface, register

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def interface() -> Generator[None, None, None]:
    register()
    with contextlib.suppress(Exception):
        yield


@pytest.fixture
def adata() -> AnnData:
    gen = np.random.default_rng()
    x = gen.random((100, 50), dtype=np.float32)
    layers = dict(a=sp.random(100, 50, rng=gen, format="csr"))
    obs = pd.DataFrame(
        dict(type=gen.integers(0, 3, size=100)),
        index="cell-" + pd.array(range(100)).astype(str),
    )
    var_grp = pd.Categorical(
        gen.integers(0, 6, size=50), categories=list(ascii_lowercase[:5])
    )
    var = pd.DataFrame(
        dict(grp=var_grp),
        index="gene-" + pd.array(range(50)).astype(str),
    )
    obsm = dict(umap=gen.random((100, 2)))
    return AnnData(x, obs, var, layers=layers, obsm=obsm, varm={})


def test_get(adata: AnnData) -> None:
    data = hv.Dataset(adata, ["obsm.umap.0"], ["obsm.umap.1", "obs.type"])
    assert data.interface is AnnDataInterface
    assert data.interface.values(data, "obs.type").equals(adata.obs["type"])
    np.testing.assert_array_equal(
        data.interface.values(data, "obsm.umap.0"), adata.obsm["umap"][:, 0]
    )
    np.testing.assert_array_equal(
        data.interface.values(data, "obsm.umap.1"), adata.obsm["umap"][:, 1]
    )


def test_plot(adata: AnnData) -> None:
    p = (
        hv.Scatter(adata, "obsm.umap.0", ["obsm.umap.1", "obs.type"])
        .opts(color="obs.type", cmap="Category20")
        .hist()
    )
