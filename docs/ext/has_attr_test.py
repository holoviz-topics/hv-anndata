"""Extension adding a jinja2 filter that tests if an object has an attribute.

See <https://jinja.palletsprojects.com/en/3.0.x/api/#custom-tests>.
"""

from __future__ import annotations

from inspect import get_annotations
from typing import TYPE_CHECKING

from jinja2.defaults import DEFAULT_NAMESPACE
from jinja2.utils import import_string

if TYPE_CHECKING:
    from sphinx.application import Sphinx


def has_member(obj_path: str, attr: str) -> bool:
    """Test if an object has an attribute."""
    obj = import_string(obj_path)
    return hasattr(obj, attr) or attr in get_annotations(obj)


def is_inherited(obj_path: str, attr: str) -> bool:
    """Test if an object attribute is inherited."""
    obj = import_string(obj_path)
    typ = obj if isinstance(obj, type) else type(obj)
    if getattr(getattr(typ, attr, None), "__override__", False):
        return False  # we’re explicitly overriding it
    return any(
        hasattr(cls, attr) or attr in get_annotations(cls) for cls in obj.mro()[1:]
    )


def setup(app: Sphinx) -> None:
    """App setup hook."""
    del app
    DEFAULT_NAMESPACE["has_member"] = has_member
    DEFAULT_NAMESPACE["is_inherited"] = is_inherited
