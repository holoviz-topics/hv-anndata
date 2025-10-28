"""Nbsite’s paramdoc extension.

Adapted from <https://github.com/holoviz-dev/nbsite/blob/main/nbsite/paramdoc.py>

See <https://github.com/holoviz/param/issues/197>
"""

from __future__ import annotations

import inspect
from contextlib import suppress
from functools import partial
from typing import TYPE_CHECKING

import param
from param.parameterized import label_formatter

if TYPE_CHECKING:
    from typing import Literal

    from sphinx.application import Sphinx

    class _Options:
        inherited_members: bool
        undoc_members: bool
        show_inheritance: bool
        no_index: bool

        members: list[str]
        # there are more


param.parameterized.docstring_signature = False
param.parameterized.docstring_describe_params = False

# Parameter attributes which are never shown
IGNORED_ATTRS = [
    "precedence",
    "check_on_set",
    "instantiate",
    "pickle_default_value",
    "watchers",
    "compute_default_fn",
    "doc",
    "owner",
    "per_instance",
    "is_instance",
    "name",
    "time_fn",
    "time_dependent",
    "rx",
]

# Default parameter attribute values (value not shown if it matches defaults)
DEFAULT_VALUES = {
    "allow_None": False,
    "readonly": False,
    "constant": False,
    "allow_refs": False,
    "nested_refs": False,
}


def param_formatter(
    app: Sphinx,  # noqa: ARG001
    what: Literal["module", "class", "exception", "function", "method", "attribute"],
    name: str,  # noqa: ARG001
    obj: object,
    options: _Options,  # noqa: ARG001
    lines: list[str],
) -> None:
    """Format parameter documentation."""
    if what != "class" or not isinstance(
        obj, param.parameterized.ParameterizedMetaclass
    ):
        return

    parameters = ["name"]
    mro = obj.mro()[::-1]
    inherited = []
    for cls in mro[:-1]:
        if not issubclass(cls, param.Parameterized) or cls is param.Parameterized:
            continue
        cls_params = [
            p for p in cls.param if p not in parameters and cls.param[p] == obj.param[p]
        ]
        if not cls_params:
            continue
        parameters += cls_params
        cname = cls.__name__
        module = cls.__module__
        inherited.extend([
            "",
            f"    :class:`{module}.{cname}`: {', '.join(cls_params)}",
        ])

    params = [p for p in obj.param if p not in parameters]
    if params:
        lines.extend(["", ".. rubric:: Parameter Definitions", ""])

    if inherited:
        lines.extend(["Parameters inherited from: ", *inherited])

    for child in params:
        lines.extend(_format_child(child, obj))


def _format_child(
    child: str, obj: param.parameterized.ParameterizedMetaclass
) -> list[str]:
    if child in ["print_level", "name"]:
        return []
    pobj = obj.param[child]
    label = label_formatter(pobj.name)
    doc = pobj.doc or ""
    members = inspect.getmembers(pobj)
    params_str = ""
    for name, value in members:
        is_default = False
        with suppress(Exception):
            is_default = bool(DEFAULT_VALUES.get(name) == value)
        skip = (
            name.startswith("_")
            or name in IGNORED_ATTRS
            or inspect.ismethod(value)
            or inspect.isfunction(value)
            or value is None
            or is_default
            or (name == "label" and pobj.label != label)
        )
        if not skip:
            params_str += f"{name}={value!r}, "
    params_str = params_str[:-2]
    ptype = pobj.__class__.__name__
    return [
        "",
        f"``{child} = {ptype}({params_str if params_str.lstrip() else ''})``",
        f"    {doc}",
    ]


def param_skip(
    app: Sphinx,  # noqa: ARG001
    what: Literal["module", "class", "exception", "function", "method", "attribute"],
    name: str,  # noqa: ARG001
    obj: object,
    skip: bool,  # noqa: FBT001
    options: _Options,
) -> bool | None:
    """Skip undocumentable parameters."""
    if what == "class" and not skip:
        return (
            getattr(obj, "__qualname__", "").startswith("Parameters.deprecate")
            or (
                isinstance(obj, partial)
                and bool(obj.args)
                and isinstance(obj.args[0], param.Parameterized)
            )
            or (
                getattr(obj, "__qualname__", "").startswith("Parameterized.")
                and getattr(obj, "__class__", str).__name__ == "function"
            )
        )
    if (
        what == "module"
        and not skip
        and isinstance(obj, type)
        and issubclass(obj, param.Parameterized)
    ):
        # HACK: Sphinx incorrectly labels this as a module level discovery  # noqa: E501, FIX004
        #       We abuse this skip callback to exclude parameters and
        #       include all methods
        members = [
            member
            for member in dir(obj)
            if not member.startswith("_") and member not in obj.param
        ]
        if isinstance(options.members, list):
            options.members += members
        else:
            options.members = members
        return skip

    return None


def setup(app: Sphinx) -> None:
    """Set extension up for sphinx."""
    app.connect("autodoc-process-docstring", param_formatter)
    app.connect("autodoc-skip-member", param_skip)
