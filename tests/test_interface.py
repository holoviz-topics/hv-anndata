"""Test anndata interface."""

from __future__ import annotations

import contextlib
from string import ascii_lowercase
from typing import TYPE_CHECKING, Any, Literal

import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData
from holoviews.core.data.interface import DataError

from hv_anndata.interface import ACCESSOR as A
from hv_anndata.interface import AnnDataGriddedInterface, AnnDataInterface, register

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


PATHS: list[
    tuple[AdPath, Callable[[AnnData], np.ndarray | sp.coo_array | pd.Series]]
] = [
    (A[:, "gene-3"], lambda ad: ad[:, "gene-3"].X.flatten()),
    (A["cell-5", :], lambda ad: ad["cell-5"].X.flatten()),
    (A.obs["type"], lambda ad: ad.obs["type"]),
    (A.obs.index, lambda ad: ad.obs.index.values),
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
    (A.varp["cons"][46, :], lambda ad: ad.varp["cons"][46, :].toarray()),
    (A.varp["cons"][:, 46], lambda ad: ad.varp["cons"][:, 46].toarray()),
]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        pytest.param(n, f, id=str(n).replace("slice(None, None, None)", ":"))
        for n, f in PATHS
    ],
)
def test_get(
    adata: AnnData,
    path: AdPath,
    expected: Callable[[AnnData], np.ndarray | sp.coo_array | pd.Series],
) -> None:
    data = hv.Dataset(adata, [path])
    assert data.interface is AnnDataInterface
    vals = data.interface.values(data, path, keep_index=True)
    if isinstance(vals, np.ndarray):
        np.testing.assert_array_equal(vals, expected(adata), strict=True)
    else:  # pragma: no cover
        pytest.fail(f"Unexpected return type {type(vals)}")


@pytest.mark.parametrize(
    ("sel_args", "sel_kw"),
    [
        pytest.param(({A.obs["type"]: 0},), {}, id="dict"),
        pytest.param((), {"obs.type": 0}, id="kwargs"),
        pytest.param(
            (hv.dim(A.obs["type"]) == 0,),
            {},
            id="expression",
            marks=pytest.mark.xfail(reason="Not implemented"),
        ),
    ],
)
def test_select(
    adata: AnnData, sel_args: tuple[Any, ...], sel_kw: dict[str, Any]
) -> None:
    data = hv.Dataset(adata, A.obsm["umap"][0], [A.obsm["umap"][1], A.obs["type"]])
    assert data.interface is AnnDataInterface
    ds_sel = data.select(*sel_args, **sel_kw)

    adata_subset = ds_sel.data
    assert isinstance(adata_subset, AnnData)
    pd.testing.assert_index_equal(
        adata_subset.obs_names, adata[adata.obs["type"] == 0].obs_names
    )


@pytest.mark.parametrize(
    ("iface", "vdims", "err_msg_pat"),
    [
        pytest.param("tab", [A[:, :]], r"cannot handle gridded", id="tab_x_2d"),
        pytest.param("grid", [A[:, "3"]], None, id="grid_x_1d"),  # can be 2D as well
        pytest.param("grid", [A.obs["x"]], r"cannot handle tabular", id="grid_obs"),
    ],
)
def test_init_errors(
    iface: Literal["tab", "grid"], vdims: list[AdPath], err_msg_pat: str
) -> None:
    cls = AnnDataInterface if iface == "tab" else AnnDataGriddedInterface
    adata = AnnData(np.zeros((10, 10)), dict(x=range(10)))
    with (
        contextlib.nullcontext()
        if err_msg_pat is None
        else pytest.raises(ValueError, match=err_msg_pat)
    ):
        cls.init(hv.Element, adata, [], vdims)


@pytest.mark.parametrize(
    ("kdims", "vdims", "err_msg_pat"),
    [
        pytest.param(
            [A.obs.index, A.var.index],
            [A.obs["x"]],
            r"either.*obs.*or.*var",
            id="grid_obs",
        ),
        pytest.param([A.obs.index, A.var.index], [A[:, "3"]], None, id="grid_x_1d"),
        # TODO: other errors  # noqa: TD003
    ],
)
def test_axes_errors(
    kdims: list[AdPath], vdims: list[AdPath], err_msg_pat: str
) -> None:
    adata = AnnData(np.zeros((10, 10)), dict(x=range(10)))
    with (
        contextlib.nullcontext()
        if err_msg_pat is None
        else pytest.raises(DataError, match=err_msg_pat)
    ):
        # Dataset.__init__ calls self.interface.axes after initializing the interface
        hv.Dataset(adata, kdims, vdims)


"""
def test_plot(adata: AnnData) -> None:
    p = (
        hv.Scatter(adata, "obsm.umap.0", ["obsm.umap.1", "obs.type"])
        .opts(color="obs.type", cmap="Category20")
        .hist()
    )
"""
