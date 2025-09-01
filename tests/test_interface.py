"""Test anndata interface."""

from __future__ import annotations

import contextlib
from string import ascii_lowercase
from typing import TYPE_CHECKING

import holoviews as hv
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData
from holoviews.core.data.interface import DataError

from hv_anndata.interface import ACCESSOR as A
from hv_anndata.interface import AnnDataGriddedInterface, AnnDataInterface, register

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from typing import Any, Literal

    from numpy.typing import ArrayLike

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


def test_get_values_table(
    request: pytest.FixtureRequest,
    adata: AnnData,
    ad_path: AdPath,
    ad_expected: Callable[[AnnData], np.ndarray | sp.coo_array | pd.Series],
) -> None:
    if ad_path.axes == {"obs", "var"}:
        request.applymarker("xfail")
    data = hv.Dataset(adata, [ad_path])

    assert data.interface is AnnDataInterface
    vals = data.interface.values(data, ad_path, keep_index=True)

    if isinstance(vals, np.ndarray):
        np.testing.assert_array_equal(vals, ad_expected(adata), strict=True)
    else:  # pragma: no cover
        pytest.fail(f"Unexpected return type {type(vals)}")


@pytest.mark.parametrize("expanded", [True, False], ids=["expanded", "normal"])
@pytest.mark.parametrize("flat", [True, False], ids=["flat", "nested"])
def test_get_values_grid(
    *,
    adata: AnnData,
    ad_path: AdPath,
    expanded: bool,
    flat: bool,
    ad_expected: Callable[[AnnData], np.ndarray | sp.coo_array | pd.Series],
) -> None:
    data = hv.Dataset(adata, [A.obs.index, A.var.index], [A[:, :], ad_path])
    assert data.interface is AnnDataGriddedInterface
    # prepare expected array
    expected = ad_expected(adata)
    if isinstance(expected, pd.Series):
        expected = expected.values
    if not isinstance(expected, np.ndarray):
        pytest.fail(f"Unexpected return type {type(expected)}")
    if expanded:
        if ad_path.axes == {"var"}:
            expected = np.broadcast_to(expected, (adata.n_obs, len(expected)))
        elif ad_path.axes == {"obs"}:
            expected = np.broadcast_to(expected, (adata.n_vars, len(expected))).T
        else:
            assert ad_path.axes == {"var", "obs"}
    if flat:
        expected = expected.flatten()

    # get values
    vals = data.interface.values(
        data, ad_path, expanded=expanded, flat=flat, keep_index=False
    )

    # compare
    if not isinstance(vals, np.ndarray):
        pytest.fail(f"Unexpected return type {type(vals)}")
    np.testing.assert_array_equal(vals, expected, strict=True)


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
        pytest.param("grid", [A[:, "3"]], r"cannot handle tabular", id="grid_x_1d"),
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
        pytest.param(
            [A.obs.index, A.var.index],
            [A[:, "3"]],
            r"either.*obs.*or.*var",
            id="grid_x_1d",
        ),
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


@pytest.mark.parametrize(
    "transposed", [True, False], ids=["transposed", "not_transposed"]
)
@pytest.mark.parametrize(
    ("v", "dim", "expected"),
    [
        pytest.param("p", A[:, :], np.arange(8).reshape((2, 4)), id="p-tabular"),
        pytest.param("p", A.obs.index, [["0"] * 4, ["1"] * 4], id="p-obs_names"),
        pytest.param("p", A.obs["x"], [["a"] * 4, ["b"] * 4], id="p-obs[x]"),
        pytest.param("p", A.var.index, [list("0123")] * 2, id="p-var_names"),
        pytest.param("p", A.var["y"], [list("ABCD")] * 2, id="p-var[y]"),
        pytest.param("l", A[:, :], np.arange(8).reshape((4, 2)), id="l-tabular"),
        pytest.param("l", A.obs.index, [[c] * 2 for c in "0123"], id="l-obs_names"),
        pytest.param("l", A.obs["x"], [[c] * 2 for c in "abcd"], id="l-obs[x]"),
        pytest.param("l", A.var.index, [["0", "1"]] * 4, id="l-var_names"),
        pytest.param("l", A.var["y"], [["A", "B"]] * 4, id="l-var[y]"),
    ],
)
def test_gridded_ax(
    *, v: Literal["p", "l"], dim: AdPath, transposed: bool, expected: ArrayLike
) -> None:
    adata = (
        AnnData(
            np.arange(8).reshape((2, 4)),
            dict(x=["a", "b"]),
            dict(y=["A", "B", "C", "D"]),
        )
        if v == "p"
        else AnnData(
            np.arange(8).reshape((4, 2)),
            dict(x=["a", "b", "c", "d"]),
            dict(y=["A", "B"]),
        )
    )
    kdims = [A.var.index, A.obs.index] if transposed else [A.obs.index, A.var.index]
    ds = hv.Dataset(adata, kdims=kdims, vdims=[A[:, :], A.obs["x"], A.var["y"]])

    values = ds.dimension_values(dim, flat=False)

    assert values.shape == (adata.shape[::-1] if transposed else adata.shape)
    npt.assert_equal(values, np.transpose(expected) if transposed else expected)


"""
def test_plot(adata: AnnData) -> None:
    p = (
        hv.Scatter(adata, "obsm.umap.0", ["obsm.umap.1", "obs.type"])
        .opts(color="obs.type", cmap="Category20")
        .hist()
    )
"""
