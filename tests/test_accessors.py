from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from hv_anndata.interface import ACCESSOR as A

if TYPE_CHECKING:
    from collections.abc import Callable

    from hv_anndata.accessors import AdPath


def test_repr(ad_path: AdPath) -> None:
    from hv_anndata.interface import ACCESSOR as A  # noqa: PLC0415

    assert repr(ad_path) == str(ad_path)
    assert repr(ad_path)[:2] in {"A.", "A["}
    assert eval(repr(ad_path)) == ad_path  # noqa: S307
    del A


@pytest.mark.parametrize(
    "mk_path",
    [
        pytest.param(lambda: A.obs[1], id="obs-nostr"),
        pytest.param(lambda: A[:3, :], id="x-partslice"),
        pytest.param(lambda: A[:, b""], id="x-nostr"),
        pytest.param(lambda: A.obsm[0], id="obsm-nostr"),
        pytest.param(lambda: A.obsm["a"][:3, 0], id="obsm-partslice"),
        pytest.param(lambda: A.obsm["a"]["b"], id="obsm-noint"),
        pytest.param(lambda: A.varp[0], id="varp-nostr"),
        # TODO: pytest.param(lambda: A.varp["x"][...], id="varp-noint"),  # noqa: TD003
    ],
)
def test_invalid(mk_path: Callable[[], AdPath]) -> None:
    with pytest.raises((ValueError, TypeError)):
        mk_path()
