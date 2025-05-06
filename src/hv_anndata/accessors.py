from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar, overload

import numpy as np
from holoviews.core.dimension import Dimension

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd
    from anndata import AnnData
    from numpy.typing import NDArray

    # full slices: e.g. a[:, 5] or a[18, :]
    Idx = TypeVar("Idx", int, str)
    Idx2D = tuple[Idx | slice, Idx | slice]


class AdPath(Dimension):
    _repr: str
    _func: Callable[[AnnData], pd.api.extensions.ExtensionArray | NDArray[Any]]

    def __init__(
        self, _repr: str, func: Callable[[AnnData], Any], /, **params: object
    ) -> None:
        super().__init__(_repr, **params)
        self._repr = _repr
        self._func = func

    def __repr__(self) -> str:
        return self._repr.replace("slice(None, None, None)", ":")  # TODO: prettier

    def __hash__(self) -> int:
        return hash(self._repr)

    def __call__(
        self, adata: AnnData
    ) -> pd.api.extensions.ExtensionArray | NDArray[Any]:
        return self._func(adata)

    def __eq__(self, dim: object) -> bool:
        # shortcut if label, number, or so matches
        if super().__eq__(dim):
            return True
        # try to resolve
        if isinstance(dim, str) and (dim := AdAc.resolve(dim, strict=False)) is None:
            return False
        # if dim is a non-matching dimension (e.g. from a string), convert
        if isinstance(dim, Dimension):
            if (
                not isinstance(dim, AdPath)
                and (dim := AdAc.from_dimension(dim, strict=False)) is None
            ):
                return False
            # dim is an AdPath, check equality
            return hash(self) == hash(dim)
        # some unknown type
        return False

    def isin(self, adata: AnnData) -> bool:
        try:
            self(adata)
        except (IndexError, KeyError):
            return False
        return True


@dataclass(frozen=True)
class LayerAcc:
    def __getitem__(self, k: str) -> LayerVecAcc:
        return LayerVecAcc(k)


@dataclass(frozen=True)
class LayerVecAcc:
    k: str

    def __getitem__(self, i: Idx2D[str]) -> AdPath:
        def get(ad: AnnData) -> pd.api.extensions.ExtensionArray | NDArray[Any]:
            return np.asarray(ad[i].layers[self.k])  # TODO: pandas

        return AdPath(f"A.layers[{self.k!r}][{i[0]!r}, {i[1]!r}]", get)


@dataclass(frozen=True)
class MetaAcc:
    ax: Literal["obs", "var"]

    def __getitem__(self, k: str) -> AdPath:
        def get(ad: AnnData) -> pd.api.extensions.ExtensionArray | NDArray[Any]:
            if k == "index":
                return getattr(ad, self.ax).index
            return getattr(ad, self.ax)[k]

        return AdPath(f"A.{self.ax}[{k!r}]", get)


@dataclass(frozen=True)
class MultiAcc:
    ax: Literal["obsm", "varm"]

    def __getitem__(self, k: str) -> MultiVecAcc:
        return MultiVecAcc(self.ax, k)


@dataclass(frozen=True)
class MultiVecAcc:
    ax: Literal["obsm", "varm"]
    k: str

    def __getitem__(self, i: int | tuple[slice, int]) -> AdPath:
        if isinstance(i, tuple):
            if i[0] != slice(None):
                msg = f"Unsupported slice {i!r}"
                raise ValueError(msg)
            i = i[1]

        def get(ad: AnnData) -> pd.api.extensions.ExtensionArray | NDArray[Any]:
            return getattr(ad, self.ax)[self.k][:, i]

        return AdPath(f"A.{self.ax}[{self.k!r}][:, {i!r}]", get)


@dataclass(frozen=True)
class GraphAcc:
    ax: Literal["obsp", "varp"]

    def __getitem__(self, k: str) -> GraphVecAcc:
        return GraphVecAcc(self.ax, k)


@dataclass(frozen=True)
class GraphVecAcc:
    ax: Literal["obsp", "varp"]
    k: str

    def __getitem__(self, i: Idx2D[int]) -> AdPath:
        def get(ad: AnnData) -> pd.api.extensions.ExtensionArray | NDArray[Any]:
            return getattr(ad, self.ax)[self.k][i]

        return AdPath(f"A.{self.ax}[{self.k!r}][{i[0]!r}, {i[1]!r}]", get)


@dataclass(frozen=True)
class AdAc:
    ATTRS: ClassVar = frozenset(
        {"layers", "obs", "var", "obsm", "varm", "obsp", "varp"}
    )

    layers: ClassVar = LayerAcc()
    obs: ClassVar = MetaAcc("obs")
    var: ClassVar = MetaAcc("var")
    obsm: ClassVar = MultiAcc("obsm")
    varm: ClassVar = MultiAcc("varm")
    obsp: ClassVar = GraphAcc("obsp")
    varp: ClassVar = GraphAcc("varp")

    def __getitem__(self, i: Idx2D[str]) -> AdPath:
        def get(ad: AnnData) -> pd.api.extensions.ExtensionArray | NDArray[Any]:
            return np.asarray(ad[i].X)  # TODO: pandas, sparse, …

        return AdPath(f"A[{i[0]!r}, {i[1]!r}]", get)

    @overload
    @classmethod
    def from_dimension(
        cls, dim: Dimension, *, strict: Literal[True] = True
    ) -> AdPath: ...
    @overload
    @classmethod
    def from_dimension(
        cls, dim: Dimension, *, strict: Literal[False]
    ) -> AdPath | None: ...

    @classmethod
    def from_dimension(cls, dim: Dimension, *, strict: bool = True) -> AdPath | None:
        """Create accessor from another dimension."""
        if TYPE_CHECKING:
            assert isinstance(dim.name, str)

        if isinstance(dim, AdPath):
            return dim
        if (rv := AdAc.resolve(dim.name, strict=strict)) is None:
            return None
        if dim.name != dim.label:
            rv.label = dim.label
        return rv

    @overload
    @classmethod
    def resolve(cls, spec: str, *, strict: Literal[True] = True) -> AdPath: ...
    @overload
    @classmethod
    def resolve(cls, spec: str, *, strict: Literal[False]) -> AdPath | None: ...

    @classmethod
    def resolve(cls, spec: str, *, strict: bool = True) -> AdPath | None:
        """Create accessor from string."""
        if not strict:
            try:
                cls.resolve(spec)
            except ValueError:
                return None

        if "." not in spec:
            msg = f"Cannot parse accessor {spec!r}"
            raise ValueError(msg)
        acc, rest = spec.split(".", 1)
        if acc not in cls.ATTRS:
            return cls()[_parse_idx_2d(acc, rest, str)]
        match getattr(cls(), acc):
            case LayerAcc() as layers:
                if m := re.fullmatch(r"([^\[]+)\[([^,]+),\s?([^\]]+)\]", rest):
                    layer, i, j = m.groups("")  # "" just for typing
                    return layers[layer][_parse_idx_2d(i, j, str)]
                msg = (
                    f"Cannot parse layer accessor {rest!r}: "
                    "should be `name[i,:]` or `name[:,j]`"
                )
            case MetaAcc() as meta:
                return meta[rest]
            case MultiAcc() as multi:
                if m := re.fullmatch(r"([^.]+)\.([\d_]+)", rest):
                    key, i = m.groups("")  # "" just for typing
                    return multi[key][int(i)]
                msg = f"Cannot parse multi accessor {rest!r}: should be `name.i`"
            case GraphAcc():
                msg = "TODO"
                raise NotImplementedError(msg)
            case AdPath():
                msg = "TODO"
                raise NotImplementedError(msg)
            case _:
                msg = f"Unhandled accessor {spec!r}. This is a bug!"
                raise AssertionError(msg)
        raise ValueError(msg)


def _parse_idx_2d(i: str, j: str, cls: type[Idx]) -> Idx2D[Idx]:
    match i, j:
        case _, ":":
            return cls(0), slice(None)
        case ":", _:
            return slice(None), cls(0)
        case _:
            msg = f"Unknown indices {i!r}, {j!r}"
            raise ValueError(msg)
