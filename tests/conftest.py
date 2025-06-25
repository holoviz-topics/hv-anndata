"""Conftest."""

from __future__ import annotations

from typing import TYPE_CHECKING

import holoviews as hv
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


def _plotting_backend(backend: str) -> None:
    pytest.importorskip(backend)
    if not hv.extension._loaded:
        hv.extension(backend)
    hv.renderer(backend)
    curent_backend = hv.Store.current_backend
    hv.Store.set_current_backend(backend)
    yield
    hv.Store.set_current_backend(curent_backend)


@pytest.fixture
def bokeh_backend() -> Iterator[None]:
    yield from _plotting_backend("bokeh")


@pytest.fixture
def mpl_backend() -> Iterator[None]:
    yield from _plotting_backend("matplotlib")
