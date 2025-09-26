from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from hv_anndata import ACCESSOR as A
from hv_anndata.accessors import AdAc

if TYPE_CHECKING:
    from collections.abc import Callable, Collection
    from typing import Literal

    from hv_anndata.accessors import AdPath


def test_repr(ad_path: AdPath) -> None:
    from hv_anndata import ACCESSOR as A  # noqa: PLC0415

    assert repr(ad_path) == str(ad_path)
    assert repr(ad_path)[:2] in {"A.", "A["}
    assert eval(repr(ad_path)) == ad_path  # noqa: S307
    del A


@pytest.mark.parametrize(
    ("ad_path", "axes"),
    [
        pytest.param(A[:, :], {"obs", "var"}, id="x"),
        pytest.param(A.layers["y"][:, :], {"obs", "var"}, id="layer"),
        # selecting one obs gives a vector along the var axis:
        pytest.param(A.layers["y"]["c", :], {"var"}, id="layer-obs"),
        pytest.param(A.var["a"], {"var"}, id="var"),
        pytest.param(A.obsm["c"][:, 0], {"obs"}, id="obsm"),
        pytest.param(A.varp["d"][:, :], ("var", "var"), id="varp"),
        pytest.param(A.varp["d"][:, "c2"], {"var"}, id="varp-col"),
    ],
)
def test_axes(ad_path: AdPath, axes: Collection[Literal["obs", "var"]]) -> None:
    assert ad_path.axes == axes


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        # TODO: pytest.param("???", A[:, :], id="x"),  # noqa: TD003
        pytest.param("layers.y[c,:]", A.layers["y"]["c", :], id="layer-obs"),
        pytest.param("layers.y[:,g]", A.layers["y"][:, "g"], id="layer-var"),
        pytest.param("obs.a", A.obs["a"], id="obs"),
        pytest.param("var.b", A.var["b"], id="var"),
        pytest.param("obsm.c.0", A.obsm["c"][:, 0], id="obsm"),
        pytest.param("varm.d.1", A.varm["d"][:, 1], id="varm"),
        pytest.param("obsp.g[c1,:]", A.obsp["d"]["c1", :], id="obsp"),
        pytest.param("obsp.g[:,c2]", A.obsp["d"][:, "c2"], id="varp"),
    ],
)
def test_resolve(spec: str, expected: AdPath) -> None:
    try:
        assert AdAc.resolve(spec) == expected
        assert spec == expected
        assert expected != ""
    except NotImplementedError:
        pytest.xfail("not implemented")


@pytest.mark.parametrize(
    "mk_path",
    [
        pytest.param(lambda: A[:3, :], id="x-partslice"),
        pytest.param(lambda: A[:, b""], id="x-nostr"),
        pytest.param(lambda: A.layers[1], id="layer-nostr"),
        pytest.param(lambda: A.layers["a"][:3, :], id="layer-partslice"),
        pytest.param(lambda: A.layers["a"][:, b""], id="layer-nostr"),
        pytest.param(lambda: A.obs[1], id="obs-nostr"),
        pytest.param(lambda: A.obsm[0], id="obsm-nostr"),
        pytest.param(lambda: A.obsm["a"][:3, 0], id="obsm-partslice"),
        pytest.param(lambda: A.obsm["a"]["b"], id="obsm-noint"),
        pytest.param(lambda: A.varp[0], id="varp-nostr"),
        pytest.param(lambda: A.varp["x"][0, :], id="varp-nostr-inner"),
    ],
)
def test_invalid(mk_path: Callable[[], AdPath]) -> None:
    with pytest.raises((ValueError, TypeError)):
        mk_path()


@pytest.mark.parametrize(
    ("expr", "expanded"),
    [
        pytest.param(A.obs[["a", "b"]], [A.obs["a"], A.obs["b"]], id="obs"),
        pytest.param(A.var[["x", "y"]], [A.var["x"], A.var["y"]], id="var"),
    ],
)
def test_special(expr: AdPath, expanded: object) -> None:
    """Some paths have shortcuts for convenience."""
    assert expr == expanded
