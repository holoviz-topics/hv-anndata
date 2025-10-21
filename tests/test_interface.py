"""Test anndata interface."""

from __future__ import annotations

import contextlib
import re
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

from hv_anndata import ACCESSOR as A
from hv_anndata.interface import (
    AnnDataGriddedInterface,
    AnnDataInterface,
    register,
    unregister,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import Any, Literal

    from numpy.typing import ArrayLike

    from hv_anndata.accessors import AdPath


@pytest.fixture(autouse=True)
def _registered_interface() -> Iterator[None]:
    register()
    with contextlib.suppress(Exception):
        yield
    unregister()


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


@pytest.mark.parametrize(
    ("hv_obj", "axes_or_err"),
    [
        pytest.param(
            lambda ad: hv.Violin(ad, vdims=[A.obs["type"]]), ("obs",), id="violin"
        ),
        pytest.param(
            lambda ad: hv.Scatter(ad, A.obsm["umap"][0], A.obsm["umap"][1]),
            ("obs",),
            id="scatter",
        ),
        pytest.param(
            lambda ad: hv.HeatMap(ad, [A.obs.index, A.var.index], A[:, :]),
            ("obs", "var"),
            id="heatmap-X",
        ),
        # TODO: obsm heatmap?  # noqa: TD003
        pytest.param(
            lambda ad: hv.HeatMap(ad, [A.var.index, A.var.index], A.varp["cons"][:, :]),
            ("var", "var"),
            id="heatmap-varp",
        ),
        # errors
        pytest.param(
            lambda ad: hv.Scatter(ad, A.obsm["umap"][0], A.var.index),
            DataError(
                "AnnData Dataset in tabular mode must reference data along either the "
                "obs or the var axis, not both."
            ),
            id="error-scatter-1D",
        ),
        pytest.param(
            lambda ad: hv.HeatMap(ad, [A.obs.index, A.obs.index], A.varp["cons"][:, :]),
            DataError("AnnData Dataset in gridded mode vdim axes must match kdims."),
            id="error-heatmap-2D",
        ),
    ],
)
def test_interface_selection(
    adata: AnnData,
    hv_obj: Callable[[AnnData], hv.Dataset],
    axes_or_err: Exception | list[Literal["obs", "var"]],
) -> None:
    if isinstance(axes_or_err, Exception):
        with pytest.raises(type(axes_or_err), match=re.escape(str(axes_or_err))):
            hv_obj(adata)
        return
    iface_expected = {1: AnnDataInterface, 2: AnnDataGriddedInterface}[len(axes_or_err)]

    dataset = hv_obj(adata)
    assert issubclass(dataset.interface, AnnDataInterface)
    assert dataset.interface is iface_expected
    assert dataset.interface.axes(dataset) == axes_or_err


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

    if not isinstance(vals, np.ndarray):  # pragma: no cover
        pytest.fail(f"Unexpected return type {type(vals)}")
    np.testing.assert_array_equal(vals, ad_expected(adata), strict=True)


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
    data = hv.Dataset(adata, [A.var.index, A.obs.index], [A[:, :], ad_path])
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
    if flat:  # see comment in AnnDataGriddedInterface.values()
        expected = expected.T.flatten()

    vals = None
    with (
        pytest.raises(ValueError, match=r"value dimension.*must cover")
        if len(ad_path.axes) == 2 and not expanded
        else contextlib.nullcontext()
    ):
        vals = data.interface.values(
            data, ad_path, expanded=expanded, flat=flat, keep_index=False
        )
    if vals is not None:
        if not isinstance(vals, np.ndarray):
            pytest.fail(f"Unexpected return type {type(vals)}")
        np.testing.assert_array_equal(vals, expected, strict=True)


@pytest.mark.parametrize(
    ("sel_args", "sel_kw"),
    [
        *(
            pytest.param(*mk_sel(zero), id=f"{sel_name}-{type(zero).__name__}")
            for sel_name, mk_sel in dict(
                dict=lambda i: (({A.obs["type"]: i},), {}),
                kwargs=lambda i: ((), {"obs.type": i}),
            ).items()
            for zero in [0, slice(0, 1), {0}, [0], lambda vs: vs == 0]
        ),
        # TODO: https://github.com/holoviz-topics/hv-anndata/issues/109
        # 1. pytest.param(hv.dim(A.obs["type"]) == 0, {}, id="expression-eq"),
        # 2. pytest.param(hv.dim(A.obs["type"]).isin([0]), {}, id="expression-isin"),
        pytest.param(
            # TODO: actually figure out how selection_specs work  # noqa: TD003
            (),
            dict(selection_specs=[hv.Dataset]),
            id="specs",
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


def test_select_var(adata: AnnData) -> None:
    data = hv.Dataset(adata, A.var.index, [A.var["grp"]])
    assert data.interface is AnnDataInterface
    ds_sel = data.select({A.var["grp"]: "a"})
    adata_subset = ds_sel.data
    assert isinstance(adata_subset, AnnData)
    pd.testing.assert_index_equal(
        adata_subset.var_names, adata[:, adata.var["grp"] == "a"].var_names
    )


@pytest.mark.parametrize(
    ("iface", "vdims", "err_msg_pat"),
    [
        pytest.param("tab", [A[:, :]], r"cannot handle gridded", id="tab-x_2d"),
        pytest.param("grid", [A[:, "3"]], r"cannot handle tabular", id="grid-x_1d"),
        pytest.param("grid", [A.obs["x"]], r"cannot handle tabular", id="grid-obs"),
    ],
)
def test_init_errors(
    iface: Literal["tab", "grid"], vdims: list[AdPath], err_msg_pat: str
) -> None:
    cls = AnnDataInterface if iface == "tab" else AnnDataGriddedInterface
    adata = AnnData(np.zeros((10, 10)), dict(x=range(10)))
    with pytest.raises(ValueError, match=err_msg_pat):
        cls.init(hv.Element, adata, [], vdims)


@pytest.mark.parametrize(
    ("kdims", "err_msg_pat"),
    [
        pytest.param(
            [A.obs["zzzzz"]], r"dimensions.*not found.*A.obs\['zzzzz'\]", id="missing"
        ),
        pytest.param(
            [A[:, :]], r"AnnData Dataset key dimensions must map onto one axis", id="2d"
        ),
    ],
)
def test_validate_errors(kdims: list[AdPath], err_msg_pat: str) -> None:
    adata = AnnData(np.zeros((10, 10)), dict(x=range(10)))
    with pytest.raises(DataError, match=err_msg_pat):
        hv.Dataset(adata, kdims)


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


@pytest.mark.parametrize("vdim", [A[:, :], A.obsm["umap"][0]], ids=["2d", "1d"])
def test_violin_2d(adata: AnnData, vdim: AdPath) -> None:
    """Violin plots don’t have kdims and just do stats on flattened data,
    so they should accept both a 1D and 2D vdim."""
    hv.Violin(adata, [], [vdim])


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
    kdims = [A.obs.index, A.var.index] if transposed else [A.var.index, A.obs.index]
    ds = hv.Dataset(adata, kdims=kdims, vdims=[A[:, :], A.obs["x"], A.var["y"]])
    assert ds.interface is AnnDataGriddedInterface

    values = ds.dimension_values(dim, expanded=True, flat=False)

    assert values.shape == (adata.shape[::-1] if transposed else adata.shape)
    npt.assert_equal(values, np.transpose(expected) if transposed else expected)
