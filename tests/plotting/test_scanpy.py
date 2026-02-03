from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pytest
import scanpy as sc
import scipy.sparse as sps
from anndata import AnnData
from holoviews.plotting.renderer import Renderer

from hv_anndata import ACCESSOR as A
from hv_anndata.interface import register, unregister
from hv_anndata.plotting import scanpy as pl

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import holoviews as hv
    from holoviews.core.layout import Layoutable
    from holoviews.plotting import Renderer


@pytest.fixture(autouse=True)
def interface() -> Iterator[None]:
    register()
    with contextlib.suppress(Exception):
        yield
    unregister()


@pytest.fixture(autouse=True, params=["bokeh", "matplotlib"])
def renderer(request: pytest.FixtureRequest) -> Renderer:
    match request.param:
        case "bokeh":
            return request.getfixturevalue("bokeh_renderer")
        case "matplotlib":
            return request.getfixturevalue("mpl_renderer")
    return request.param


@pytest.mark.parametrize(
    "do_plot",
    [
        pytest.param(
            lambda ad: pl.scatter(ad, A.obsm["umap"][[0, 1]]), id="scatter-obsm"
        ),
        pytest.param(
            lambda ad: pl.scatter(ad, A.obsp["distances"][["cell-0", "cell-1"], :]),
            id="scatter-obsp",
        ),
        pytest.param(pl.heatmap, id="heatmap-obsm"),
        # TODO: pytest.param(lambda ad: pl.heatmap(ad, A.obsp["distances"]), id="heatmap-obsp"),  # noqa: E501
        # https://github.com/holoviz-topics/hv-anndata/issues/111
        pytest.param(
            lambda ad: pl.violin(ad, A.obsm["umap"][0], color=A.obs.index),
            id="violin-obsm-color",
        ),
        pytest.param(
            lambda ad: pl.violin(ad, [A.obsm["umap"][0], A.obsm["umap"][1]]),
            id="violin-obsm-multi",
        ),
    ],
)
def test_basic(
    renderer: Renderer, do_plot: Callable[[AnnData], hv.Dataset | Layoutable]
) -> None:
    rng = np.random.default_rng()
    adata = AnnData(
        sps.random_array((20, 8), density=0.7, format="csr", rng=rng),
        dict(obs_names=["cell-" + str(i) for i in range(20)]),
        dict(var_names=["gene-" + str(i) for i in range(8)]),
        obsm=dict(umap=rng.random((20, 2))),
    )
    sc.pp.neighbors(adata)
    obj = do_plot(adata)
    _plot = renderer.get_plot(obj)
