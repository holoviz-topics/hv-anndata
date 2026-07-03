"""Stop Napoleon from misreading a colon in prose as a ``type: description`` field.

For attribute/data/property docstrings,
Napoleon’s ``_consume_inline_attribute`` splits the first line on the first colon
and treats the part before it as a *type*.

This extension patches it to only treat identifiers to the left of colons as types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sphinx.ext.napoleon.docstring import GoogleDocstring
from sphinx.util.typing import ExtensionMetadata

if TYPE_CHECKING:
    from sphinx.application import Sphinx


def _consume_inline_attribute(self: GoogleDocstring) -> tuple[str, list[str]]:
    line = self._lines.next()
    type_, colon, desc_ = self._partition_field_on_colon(line)
    if not colon or not desc_ or not type_.isidentifier():
        type_, desc_ = "", line
    descs = [desc_, *self._dedent(self._consume_to_end())]
    descs = self.__class__(descs, self._config).lines()
    return type_, descs


def setup(app: Sphinx) -> ExtensionMetadata:
    """Apply the Napoleon monkeypatch."""
    del app
    GoogleDocstring._consume_inline_attribute = _consume_inline_attribute  # noqa: SLF001
    return ExtensionMetadata(parallel_read_safe=True)
