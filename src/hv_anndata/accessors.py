"""Accessor classes for AnnData interface."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, TypeVar, cast, overload

import numpy as np
import scipy.sparse as sp
from holoviews.core.dimension import Dimension

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Set as AbstractSet

    import pandas as pd
    from anndata import AnnData
    from numpy.typing import NDArray

    # full slices: e.g. a[:, 5] or a[18, :]
    Idx = TypeVar("Idx", int, str)
    Idx2D = tuple[Idx | slice, Idx | slice]
    AdPathFunc = Callable[[AnnData], pd.api.extensions.ExtensionArray | NDArray[Any]]
    Axes = AbstractSet[Literal["obs", "var"]]


def _idx2axes(i: Idx2D[str]) -> set[Literal["obs", "var"]]:
    """Get along which axes the referenced vector is."""
    match i:
        case slice(), str():
            return {"obs"}
        case str(), slice():
            return {"var"}
        case slice(), slice():
            return {"obs", "var"}
        case _:
            msg = f"Invalid index: {i}"
            raise AssertionError(msg)


class AdPath(Dimension):
    """A path referencing an array in an AnnData object."""

    _repr: str
    _func: AdPathFunc
    axes: Axes

    def __init__(  # noqa: D107
        self,
        _repr: str | tuple[str, str],
        func: AdPathFunc,
        axes: Axes,
        /,
        **params: object,
    ) -> None:
        super().__init__(_repr, **params)
        self._repr = _repr[0] if isinstance(_repr, tuple) else _repr
        self._func = func
        self.axes = axes

    def __repr__(self) -> str:
        # TODO: prettier  # noqa: TD003
        return self._repr.replace("slice(None, None, None)", ":")

    def __hash__(self) -> int:
        return hash(self._repr)

    def __call__(
        self, adata: AnnData
    ) -> pd.api.extensions.ExtensionArray | NDArray[Any]:
        """Retrieve referenced array from AnnData."""
        return self._func(adata)

    def clone(
        self,
        spec: str | tuple[str, str] | None = None,
        func: AdPathFunc = None,
        axes: Axes | None = None,
        **overrides: Any,  # noqa: ANN401
    ) -> Self:
        """Clones the Dimension with new parameters.

        Derive a new Dimension that inherits existing parameters
        except for the supplied, explicit overrides

        Parameters
        ----------
        spec : tuple, optional
            Dimension tuple specification
        func : Function[AnnData, np.ndarray], optional
            Function to resolve the dimension values
            given the AnnData object.
        axes : AbstractSet[Literal["obs", "var"]], optional
            The axes represented by the Dimension
        **overrides:
            Dimension parameter overrides

        Returns
        -------
        Cloned Dimension object

        """
        settings = dict(self.param.values(), **overrides)
        func = settings.pop("func", self._func)
        axes = settings.pop("axes", self.axes)

        if spec is None:
            spec = (self.name, overrides.get("label", self.label))
        if "label" in overrides and isinstance(spec, str):
            spec = (spec, overrides["label"])
        elif "label" in overrides and isinstance(spec, tuple):
            if overrides["label"] != spec[1]:
                self.param.warning(
                    f"Using label as supplied by keyword ({overrides['label']!r}), "
                    f"ignoring tuple value {spec[1]!r}"
                )
            spec = (spec[0], overrides["label"])
        return self.__class__(
            spec,
            func,
            axes,
            **{k: v for k, v in settings.items() if k not in ["name", "label"]},
        )

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
        """Check if array is in AnnData."""
        try:
            self(adata)
        except (IndexError, KeyError):
            return False
        return True


@dataclass(frozen=True)
class LayerAcc:
    """Accessor for layers."""

    def __getitem__(self, k: str) -> LayerVecAcc:
        return LayerVecAcc(k)


@dataclass(frozen=True)
class LayerVecAcc:
    """Accessor for layer vectors."""

    k: str

    def __getitem__(self, i: Idx2D[str]) -> AdPath:
        def get(ad: AnnData) -> pd.api.extensions.ExtensionArray | NDArray[Any]:
            ver_or_mat = ad[i].layers[self.k]
            if isinstance(ver_or_mat, sp.spmatrix | sp.sparray):
                ver_or_mat = ver_or_mat.toarray().flatten()
            return ver_or_mat  # TODO: pandas  # noqa: TD003

        return AdPath(f"A.layers[{self.k!r}][{i[0]!r}, {i[1]!r}]", get, _idx2axes(i))


@dataclass(frozen=True)
class MetaAcc:
    """Accessor for metadata (obs/var)."""

    ax: Literal["obs", "var"]

    def __getitem__(self, k: str) -> AdPath:
        def get(ad: AnnData) -> pd.api.extensions.ExtensionArray | NDArray[Any]:
            if k == "index":
                return getattr(ad, self.ax).index
            return getattr(ad, self.ax)[k]

        return AdPath(f"A.{self.ax}[{k!r}]", get, {self.ax})


@dataclass(frozen=True)
class MultiAcc:
    """Accessor for multi-dimensional array containers (obsm/varm)."""

    ax: Literal["obsm", "varm"]

    def __getitem__(self, k: str) -> MultiVecAcc:
        return MultiVecAcc(self.ax, k)


@dataclass(frozen=True)
class MultiVecAcc:
    """Accessor for arrays from multi-dimensional containers (obsm/varm)."""

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

        ax = cast("Literal['obs', 'var']", self.ax[:-1])
        return AdPath(f"A.{self.ax}[{self.k!r}][:, {i!r}]", get, {ax})


@dataclass(frozen=True)
class GraphAcc:
    """Accessor for graph containers (obsp/varp)."""

    ax: Literal["obsp", "varp"]

    def __getitem__(self, k: str) -> GraphVecAcc:
        return GraphVecAcc(self.ax, k)


@dataclass(frozen=True)
class GraphVecAcc:
    """Accessor for arrays from graph containers (obsp/varp)."""

    ax: Literal["obsp", "varp"]
    k: str

    def __getitem__(self, i: Idx2D[int]) -> AdPath:
        def get(ad: AnnData) -> pd.api.extensions.ExtensionArray | NDArray[Any]:
            return getattr(ad, self.ax)[self.k][i].toarray().flatten()

        ax = cast("Literal['obs', 'var']", self.ax[:-1])
        return AdPath(f"A.{self.ax}[{self.k!r}][{i[0]!r}, {i[1]!r}]", get, {ax})


class AdAc:
    r"""Accessor singleton to create :class:`AdPath`\ s."""

    ATTRS: ClassVar = frozenset(
        {"layers", "obs", "var", "obsm", "varm", "obsp", "varp"}
    )
    _instance: ClassVar[Self]

    layers: ClassVar = LayerAcc()
    obs: ClassVar = MetaAcc("obs")
    var: ClassVar = MetaAcc("var")
    obsm: ClassVar = MultiAcc("obsm")
    varm: ClassVar = MultiAcc("varm")
    obsp: ClassVar = GraphAcc("obsp")
    varp: ClassVar = GraphAcc("varp")

    def __new__(cls) -> Self:  # noqa: D102
        if not hasattr(cls, "_instance"):
            cls._instance = object.__new__(cls)
        return cls._instance

    def __getitem__(self, i: Idx2D[str]) -> AdPath:
        def get(ad: AnnData) -> pd.api.extensions.ExtensionArray | NDArray[Any]:
            return np.asarray(ad[i].X)  # TODO: pandas, sparse, …  # noqa: TD003

        return AdPath(f"A[{i[0]!r}, {i[1]!r}]", get, _idx2axes(i))

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
        match getattr(cls(), acc, None):
            case LayerAcc() as layers:
                return _parse_path_layer(layers, rest)
            case MetaAcc() as meta:
                return meta[rest]
            case MultiAcc() as multi:
                return _parse_path_multi(multi, rest)
            case GraphAcc():
                msg = "TODO"
                raise NotImplementedError(msg)
            case AdPath():
                msg = "TODO"
                raise NotImplementedError(msg)
            case None:
                msg = (
                    f"Unknown accessor {spec!r}. "
                    "We support '{cls.ATTRS}.*' and `AdPath` instances."
                )
                raise ValueError(msg)
        msg = f"Unhandled accessor {spec!r}. This is a bug!"
        raise AssertionError(msg)


def _parse_idx_2d(i: str, j: str, cls: type[Idx]) -> Idx2D[Idx]:
    match i, j:
        case _, ":":
            return cls(0), slice(None)
        case ":", _:
            return slice(None), cls(0)
        case _:
            msg = f"Unknown indices {i!r}, {j!r}"
            raise ValueError(msg)


def _parse_path_layer(layers: LayerAcc, spec: str) -> AdPath:
    if m := re.fullmatch(r"([^\[]+)\[([^,]+),\s?([^\]]+)\]", spec):
        layer, i, j = m.groups("")  # "" just for typing
        return layers[layer][_parse_idx_2d(i, j, str)]
    msg = f"Cannot parse layer accessor {spec!r}: should be `name[i,:]` or `name[:,j]`"
    raise ValueError(msg)


def _parse_path_multi(multi: MultiAcc, spec: str) -> AdPath:
    if m := re.fullmatch(r"([^.]+)\.([\d_]+)", spec):
        key, i = m.groups("")  # "" just for typing
        return multi[key][int(i)]
    msg = f"Cannot parse multi accessor {spec!r}: should be `name.i`"
    raise ValueError(msg)
