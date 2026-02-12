"""Conftest."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING

import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

from hv_anndata import A

if TYPE_CHECKING:
    from collections.abc import Iterator

    from holoviews.plotting import Renderer

    from hv_anndata import AdDim


@contextmanager
def _renderer(backend: str) -> Iterator[Renderer]:
    pytest.importorskip(backend)
    if not hv.extension._loaded:
        hv.extension(backend)
    renderer = hv.renderer(backend)
    old_backend = hv.Store.current_backend
    hv.Store.set_current_backend(backend)
    yield renderer
    hv.Store.set_current_backend(old_backend)


@pytest.fixture
def bokeh_renderer() -> Iterator[Renderer]:
    with _renderer("bokeh") as renderer:
        yield renderer


@pytest.fixture
def mpl_renderer() -> Iterator[Renderer]:
    with _renderer("matplotlib") as renderer:
        yield renderer


type AdDimExpected = Callable[[AnnData], np.ndarray | sp.coo_array | pd.Series]

PATHS: list[tuple[AdDim, AdDimExpected]] = [
    (A.X[:, :], lambda ad: ad.X),
    (A.X[:, "gene-3"], lambda ad: ad[:, "gene-3"].X.flatten()),
    (A.X["cell-5", :], lambda ad: ad["cell-5"].X.flatten()),
    (A.obs["type"], lambda ad: ad.obs["type"]),
    (A.obs.index, lambda ad: ad.obs.index.values),
    (A.layers["a"][:, :], lambda ad: ad.layers["a"].copy().toarray()),
    (
        A.layers["a"][:, "gene-18"],
        lambda ad: ad[:, "gene-18"].layers["a"].copy().toarray().flatten(),
    ),
    (
        A.layers["a"]["cell-77", :],
        lambda ad: ad["cell-77"].layers["a"].copy().toarray().flatten(),
    ),
    (A.obsm["umap"][0], lambda ad: ad.obsm["umap"][:, 0]),
    (A.obsm["umap"][1], lambda ad: ad.obsm["umap"][:, 1]),
    (A.varp["cons"]["gene-46", :], lambda ad: ad.varp["cons"][46, :].toarray()),
    (A.varp["cons"][:, "gene-46"], lambda ad: ad.varp["cons"][:, 46].toarray()),
]


@pytest.fixture(scope="session", params=PATHS, ids=[str(p[0]) for p in PATHS])
def path_and_expected_fn(
    request: pytest.FixtureRequest,
) -> tuple[AdDim, AdDimExpected]:
    return request.param


@pytest.fixture(scope="session")
def ad_dim(path_and_expected_fn: tuple[AdDim, AdDimExpected]) -> AdDim:
    return path_and_expected_fn[0]


@pytest.fixture(scope="session")
def ad_expected(path_and_expected_fn: tuple[AdDim, AdDimExpected]) -> AdDimExpected:
    return path_and_expected_fn[1]
