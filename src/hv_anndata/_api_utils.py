from __future__ import annotations

from types import FunctionType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = ["set_module"]


def set_module[T: type | FunctionType](module: str, /) -> Callable[[T], T]:
    def decorator(obj: T, /) -> T:
        obj.__module__ = module
        return obj

    return decorator
