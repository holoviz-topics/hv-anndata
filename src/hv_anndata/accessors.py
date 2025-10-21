"""Accessor classes for AnnData interface."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast, overload

import scipy.sparse as sp
from holoviews.core.dimension import Dimension

if TYPE_CHECKING:
    from collections.abc import Callable, Collection
    from typing import Any, Literal, Self, TypeIs, TypeVar

    import pandas as pd
    from anndata import AnnData
    from numpy.typing import NDArray

    # full slices: e.g. a[:, 5] or a[18, :]
    Idx = TypeVar("Idx", int, str)
    Idx2D = tuple[Idx | slice, slice] | tuple[slice, Idx | slice]
    Idx2DList = tuple[list[Idx], slice] | tuple[slice, list[Idx]]
    AdPathFunc = Callable[[AnnData], pd.api.extensions.ExtensionArray | NDArray[Any]]
    Axes = Collection[Literal["obs", "var"]]


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
        # TODO: prettier  # noqa: TD003
        if isinstance(_repr, str):
            _repr = _repr.replace("slice(None, None, None)", ":")
        super().__init__(_repr, **params)
        self._repr = _repr[0] if isinstance(_repr, tuple) else _repr
        self._func = func
        self.axes = axes

    def __repr__(self) -> str:
        return str(self)

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
        func: AdPathFunc | None = None,
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
        if not isinstance(k, str):
            msg = f"Unsupported layer {k!r}"
            raise TypeError(msg)
        return LayerVecAcc(k)


@dataclass(frozen=True)
class LayerVecAcc:
    """Accessor for layer vectors."""

    k: str | None

    @overload
    def __getitem__(self, idx: Idx2D[str]) -> AdPath: ...
    @overload
    def __getitem__(self, idx: Idx2DList[str]) -> list[AdPath]: ...
    def __getitem__(self, idx: Idx2D[str] | Idx2DList[str]) -> AdPath | list[AdPath]:
        if _is_idx2d_list(idx):
            return [self[i] for i in _expand_idx2d_list(idx)]

        axes = _idx2axes(idx)

        def get(ad: AnnData) -> pd.api.extensions.ExtensionArray | NDArray[Any]:
            ver_or_mat = ad[idx].X if self.k is None else ad[idx].layers[self.k]
            if isinstance(ver_or_mat, sp.spmatrix | sp.sparray):
                ver_or_mat = ver_or_mat.toarray()
            # TODO: pandas  # noqa: TD003
            return ver_or_mat.flatten() if len(axes) == 1 else ver_or_mat

        sub = "" if self.k is None else f".layers[{self.k!r}]"
        return AdPath(f"A{sub}[{idx[0]!r}, {idx[1]!r}]", get, axes)


@dataclass(frozen=True)
class MetaAcc:
    """Accessor for metadata (obs/var)."""

    ax: Literal["obs", "var"]

    @property
    def index(self) -> AdPath:
        """Index accessor."""

        def get(ad: AnnData) -> pd.api.extensions.ExtensionArray | NDArray[Any]:
            return cast("pd.DataFrame", getattr(ad, self.ax)).index.values

        return AdPath(f"A.{self.ax}.index", get, {self.ax})

    @overload
    def __getitem__(self, k: str) -> AdPath: ...
    @overload
    def __getitem__(self, k: list[str]) -> list[AdPath]: ...
    def __getitem__(self, k: str | Iterable[str]) -> AdPath | list[AdPath]:
        if not isinstance(k, str) and isinstance(k, Iterable):
            return [self[i] for i in k]

        if not isinstance(k, str):
            msg = f"Unsupported {self.ax} column {k!r}"
            raise TypeError(msg)

        def get(ad: AnnData) -> pd.api.extensions.ExtensionArray | NDArray[Any]:
            return cast("pd.DataFrame", getattr(ad, self.ax))[k].values

        return AdPath(f"A.{self.ax}[{k!r}]", get, {self.ax})


@dataclass(frozen=True)
class MultiAcc:
    """Accessor for multi-dimensional array containers (obsm/varm)."""

    ax: Literal["obsm", "varm"]

    def __getitem__(self, k: str) -> MultiVecAcc:
        if not isinstance(k, str):
            msg = f"Unsupported {self.ax} key {k!r}"
            raise TypeError(msg)
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

        if not isinstance(i, int):
            msg = f"Unsupported index {i!r}"
            raise TypeError(msg)

        def get(ad: AnnData) -> pd.api.extensions.ExtensionArray | NDArray[Any]:
            return getattr(ad, self.ax)[self.k][:, i]

        ax = cast("Literal['obs', 'var']", self.ax[:-1])
        return AdPath(f"A.{self.ax}[{self.k!r}][:, {i!r}]", get, {ax})


@dataclass(frozen=True)
class GraphAcc:
    """Accessor for graph containers (obsp/varp)."""

    ax: Literal["obsp", "varp"]

    def __getitem__(self, k: str) -> GraphVecAcc:
        if not isinstance(k, str):
            msg = f"Unsupported {self.ax} key {k!r}"
            raise TypeError(msg)
        return GraphVecAcc(self.ax, k)


@dataclass(frozen=True)
class GraphVecAcc:
    """Accessor for arrays from graph containers (obsp/varp)."""

    ax: Literal["obsp", "varp"]
    k: str

    @overload
    def __getitem__(self, idx: Idx2D[str]) -> AdPath: ...
    @overload
    def __getitem__(self, idx: Idx2DList[str]) -> list[AdPath]: ...
    def __getitem__(self, idx: Idx2D[str] | Idx2DList[str]) -> AdPath | list[AdPath]:
        if _is_idx2d_list(idx):
            return [self[i] for i in _expand_idx2d_list(idx)]

        if not all(isinstance(i, str | slice) for i in idx):
            msg = f"Unsupported index {idx!r}"
            raise TypeError(msg)
        if (n_slices := sum(isinstance(i, slice) for i in idx)) not in {1, 2}:
            msg = (
                f"Unsupported index {idx!r}: "
                f"should be ['c1', :], ['c1', 'c2'], or [:, 'c2']"
            )
            raise TypeError(msg)

        def get(ad: AnnData) -> pd.api.extensions.ExtensionArray | NDArray[Any]:
            df = cast("pd.DataFrame", getattr(ad, self.ax[:-1]))
            iloc = tuple(df.index.get_loc(i) if isinstance(i, str) else i for i in idx)
            return getattr(ad, self.ax)[self.k][iloc].toarray()

        ax = cast("Literal['obs', 'var']", self.ax[:-1])
        axes: Collection[Literal["obs", "var"]] = (
            (ax,) * n_slices if n_slices > 1 else {ax}
        )
        return AdPath(f"A.{self.ax}[{self.k!r}][{idx[0]!r}, {idx[1]!r}]", get, axes)


@dataclass(frozen=True)
class AdAc(LayerVecAcc):
    r"""Accessor singleton to create :class:`AdPath`\ s."""

    k: None = None

    ATTRS: ClassVar = frozenset({
        "layers",
        "obs",
        "var",
        "obsm",
        "varm",
        "obsp",
        "varp",
    })
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
            # TODO: X  # noqa: TD003
            case LayerAcc() as layers:
                return _parse_path_layer(layers, rest)
            case MetaAcc() as meta:
                return meta[rest]
            case MultiAcc() as multi:
                return _parse_path_multi(multi, rest)
            case GraphAcc():
                msg = "TODO"
                raise NotImplementedError(msg)
            case None:  # pragma: no cover
                msg = (
                    f"Unknown accessor {spec!r}. "
                    "We support '{cls.ATTRS}.*' and `AdPath` instances."
                )
                raise ValueError(msg)
        msg = f"Unhandled accessor {spec!r}. This is a bug!"  # pragma: no cover
        raise AssertionError(msg)  # pragma: no cover


def _is_idx2d_list(idx: Idx2D[Idx] | Idx2DList[Idx]) -> TypeIs[Idx2DList[Idx]]:
    """Check if a 2D index contains a list in one of its dimensions."""
    return any(isinstance(i, list) for i in idx)


def _expand_idx2d_list(idx: Idx2D[Idx] | Idx2DList[Idx]) -> list[Idx2D[Idx]]:
    """Expand a 2D index containing a list in one of its dimensions.

    Also validates that the 2D index contains at most one list.
    """
    match idx:
        case list() as ixs, iy:
            return [(ix, iy) for ix in ixs]
        case ix, list() as iys:
            return [(ix, iy) for iy in iys]
    msg = "2D index can have at most one list: …[:, [...]] or …[[...], :]"
    raise TypeError(msg)


def _idx2axes(idx: Idx2D[str]) -> set[Literal["obs", "var"]]:
    """Get along which axes the referenced vector is."""
    for ax_idx in idx:
        if isinstance(ax_idx, str):
            continue
        if isinstance(ax_idx, slice) and ax_idx == slice(None):
            continue
        msg = (
            f"Unsupported axis index {ax_idx!r} in index {idx!r} (not `:` or a string)"
        )
        raise ValueError(msg)
    match idx:
        case slice(), str():
            return {"obs"}
        case str(), slice():
            return {"var"}
        case slice(), slice():
            return {"obs", "var"}
        case _:  # pragma: no cover
            msg = f"Invalid index: {idx}"
            raise AssertionError(msg)


def _parse_idx_2d(i: str, j: str, cls: type[Idx]) -> Idx2D[Idx]:
    match i, j:
        case _, ":":
            return cls(i), slice(None)
        case ":", _:
            return slice(None), cls(j)
        case _:  # pragma: no cover
            msg = f"Unknown indices {i!r}, {j!r}"
            raise ValueError(msg)


def _parse_path_layer(layers: LayerAcc, spec: str) -> AdPath:
    if not (
        m := re.fullmatch(r"([^\[]+)\[([^,]+),\s?([^\]]+)\]", spec)
    ):  # pragma: no cover
        msg = f"Cannot parse layer accessor {spec!r}: should be `name[i,:]`/`name[:,j]`"
        raise ValueError(msg)
    layer, i, j = m.groups("")  # "" just for typing
    return layers[layer][_parse_idx_2d(i, j, str)]


def _parse_path_multi(multi: MultiAcc, spec: str) -> AdPath:
    if not (m := re.fullmatch(r"([^.]+)\.([\d_]+)", spec)):  # pragma: no cover
        msg = f"Cannot parse multi accessor {spec!r}: should be `name.i`"
        raise ValueError(msg)
    key, i = m.groups("")  # "" just for typing
    return multi[key][int(i)]
