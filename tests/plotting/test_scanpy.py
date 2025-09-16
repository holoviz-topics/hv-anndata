from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pytest
from anndata import AnnData

from hv_anndata import ACCESSOR as A
from hv_anndata.interface import register, unregister
from hv_anndata.plotting import scanpy as pl

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def interface() -> Generator[None, None, None]:
    register()
    with contextlib.suppress(Exception):
        yield
    unregister()


@pytest.fixture(autouse=True, params=["bokeh", "matplotlib"])
def backend(request: pytest.FixtureRequest) -> str:
    match request.param:
        case "bokeh":
            request.getfixturevalue("bokeh_backend")
        case "matplotlib":
            request.getfixturevalue("mpl_backend")
    return request.param


def test_scatter() -> None:
    rng = np.random.default_rng()
    adata = AnnData(obsm=dict(umap=rng.random((20, 2))))
    pl.scatter(adata, A.obsm["umap"])
