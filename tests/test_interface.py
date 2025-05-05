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

from hv_anndata.interface import ACCESSOR as A
from hv_anndata.interface import AnnDataInterface, register

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from hv_anndata.accessors import AdPath


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
    varp = dict(cons=sp.csr_array(sp.random(50, 50, rng=gen)))
    return AnnData(x, obs, var, layers=layers, obsm=obsm, varm={}, obsp={}, varp=varp)


PATHS: list[tuple[AdPath, Callable[[AnnData], np.ndarray | pd.Series]]] = [
    (A[:, "gene-3"], lambda ad: ad[:, "gene-3"].X.flatten()),
    (A["cell-5", :], lambda ad: ad["cell-5"].X.flatten()),
    (A.obs["type"], lambda ad: ad.obs["type"]),
    (A.layers["a"][:, "gene-18"], lambda ad: ad[:, "gene-18"].layers["a"].flatten()),
    (A.layers["a"]["cell-77", :], lambda ad: ad["cell-77"].layers["a"].flatten()),
    (A.obsm["umap"][0], lambda ad: ad.obsm["umap"][:, 0]),
    (A.obsm["umap"][1], lambda ad: ad.obsm["umap"][:, 1]),
    (A.varp["cons"][46, :], lambda ad: ad.varp["varp"][46, :]),
    (A.varp["cons"][:, 46], lambda ad: ad.varp["varp"][:, 46]),
]


@pytest.mark.parametrize(
    ("path", "expected"), [pytest.param(n, f, id=str(n)) for n, f in PATHS]
)
def test_get(
    adata: AnnData, path: AdPath, expected: Callable[[AnnData], np.ndarray | pd.Series]
) -> None:
    data = hv.Dataset(adata, [path])
    assert data.interface is AnnDataInterface
    vals = data.interface.values(data, path)
    if isinstance(vals, np.ndarray):
        np.testing.assert_array_equal(vals, expected(adata))
    elif isinstance(vals, pd.Series):
        pd.testing.assert_series_equal(vals, expected(adata))
    elif isinstance(vals, pd.api.extensions.ExtensionArray):
        pd.testing.assert_extension_array_equal(vals, expected(adata))
    else:
        pytest.fail(f"Unexpected return type {type(vals)}")


"""
def test_plot(adata: AnnData) -> None:
    p = (
        hv.Scatter(adata, "obsm.umap.0", ["obsm.umap.1", "obs.type"])
        .opts(color="obs.type", cmap="Category20")
        .hist()
    )
"""
