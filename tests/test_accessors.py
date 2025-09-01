from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hv_anndata.accessors import AdPath


def test_repr(ad_path: AdPath) -> None:
    from hv_anndata.interface import ACCESSOR as A  # noqa: PLC0415

    assert repr(ad_path) == str(ad_path)
    assert repr(ad_path)[:2] in {"A.", "A["}
    assert eval(repr(ad_path)) == ad_path  # noqa: S307
    del A
