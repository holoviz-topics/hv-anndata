"""Execute the ``.. holoviews::`` example blocks in scanpy's ``plotting._v2`` doc."""

from __future__ import annotations

import inspect
import linecache
import re
import textwrap
from typing import TYPE_CHECKING

import pytest
import scanpy as sc
import scanpy.plotting._v2 as sc_pl_v2  # ruff: ignore[import-private-name]

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

#: Backends to substitute for the ``FAKE_BACKEND`` placeholder used in the examples.
BACKENDS = ["bokeh", "matplotlib", "plotly"]

WARNING_FILTERS = dict(
    umap=[r"ignore::ImportWarning"],
    heatmap=[r"default:.*Dimension\.name.*:DeprecationWarning"],
    # TODO: remove once scanpy 1.13.0.a2 is released  # ruff: ignore[missing-todo-link]
    highly_variable_genes=[r"default:.*seurat_v3.*count data.*:UserWarning"],
)

_DIRECTIVE_RE = re.compile(r"^(?P<indent>[ \t]*)\.\.[ \t]+holoviews::[ \t]*$")
_OPTION_RE = re.compile(r"^[ \t]*:(?P<name>\w+):[ \t]*(?P<value>.*)$")


def _holoviews_blocks(doc: str) -> Iterator[tuple[str, list[str] | None]]:
    """Yield ``(code, backends)`` for each ``.. holoviews::`` block in `doc`."""
    lines = doc.splitlines()
    i = 0
    while i < len(lines):
        m = _DIRECTIVE_RE.match(lines[i])
        if not m:
            i += 1
            continue
        indent = len(m["indent"])
        i += 1

        backends = None
        while i < len(lines) and (om := _OPTION_RE.match(lines[i])):
            if om["name"] == "backends":
                backends = [b.strip() for b in om["value"].split(",") if b.strip()]
            i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1

        content = []
        while i < len(lines) and (
            not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > indent
        ):
            content.append(lines[i])
            i += 1
        while content and not content[-1].strip():
            content.pop()

        yield textwrap.dedent("\n".join(content)), backends


def _make_example_test(
    name: str, blocks: list[tuple[str, list[str] | None]]
) -> Callable[[str], None]:
    allowed = None
    for _code, backends in blocks:
        if backends is None:
            continue
        allowed = set(backends) if allowed is None else allowed & set(backends)
    code_template = "\n".join(code for code, _backends in blocks)

    def test(backend: str) -> None:
        if allowed is not None and backend not in allowed:
            pytest.skip(f"{name!r} example doesn’t support the {backend!r} backend")
        code = code_template.replace("FAKE_BACKEND", repr(backend))
        filename = f"<{name}[{backend}] docstring>"
        lines = code.splitlines(keepends=True)
        linecache.cache[filename] = (len(code), None, lines, filename)
        exec(compile(code, filename, "exec"), {})  # ruff: ignore[exec-builtin]

    test.__name__ = test.__qualname__ = f"test_{name}"
    for mark in [
        pytest.mark.parametrize("backend", BACKENDS),
        *(pytest.mark.filterwarnings(w) for w in WARNING_FILTERS.get(name, ())),
    ]:
        test = mark(test)
    return test


generated = {
    f"test_{name}": _make_example_test(name, blocks)
    for name in sc_pl_v2.__all__
    if (
        blocks := list(_holoviews_blocks(inspect.getdoc(getattr(sc_pl_v2, name)) or ""))
    )
}
assert generated, "No `.. holoviews::` examples found in scanpy.plotting._v2"
globals().update(generated)


@pytest.fixture(autouse=True)
def _restore_scanpy_preset() -> Iterator[None]:
    preset = sc.settings.preset
    yield
    sc.settings.preset = preset
