from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from holoviews.core.dimension import Dimension

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd
    from anndata import AnnData
    from numpy.typing import NDArray

    # full slices: e.g. a[:, 5] or a[18, :]
    IdxInt2D = tuple[int, slice] | tuple[slice, int]
    IdxStr2D = tuple[str, slice] | tuple[slice, str]


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

    def __hash__(self):
        return hash(self._repr)

    def __call__(
        self, adata: AnnData
    ) -> pd.api.extensions.ExtensionArray | NDArray[Any]:
        return self._func(adata)

    def __eq__(self, dim: str | Dimension):
        equals = super().__eq__(dim)
        if equals or isinstance(dim, Dimension):
            return equals
        if isinstance(dim, str):
            # Ensure that selections along the var and obs dimensions match
            label = self.name.lstrip("A.").replace("['", ".").replace("']", "")
            if (
                (label.startswith("obs") and dim.startswith("obs")) or
                (label.startswith("var") and dim.startswith("var"))
            ):
                return True
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

    def __getitem__(self, i: IdxStr2D) -> AdPath:
        def get(ad: AnnData) -> pd.api.extensions.ExtensionArray | NDArray[Any]:
            return np.asarray(ad[i].layers[self.k])  # TODO: pandas

        return AdPath(f"A.layers[{self.k!r}][{i[0]!r}, {i[1]!r}]", get)


@dataclass(frozen=True)
class MetaAcc:
    ax: Literal["obs", "var"]

    def __getitem__(self, k: str) -> AdPath:
        def get(ad: AnnData) -> pd.api.extensions.ExtensionArray | NDArray[Any]:
            if k == 'index':
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

    def __getitem__(self, i: int) -> AdPath:
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

    def __getitem__(self, i: IdxInt2D) -> AdPath:
        def get(ad: AnnData) -> pd.api.extensions.ExtensionArray | NDArray[Any]:
            return getattr(ad, self.ax)[self.k][i]

        return AdPath(f"A.{self.ax}[{self.k!r}][{i[0]!r}, {i[1]!r}]", get)


@dataclass(frozen=True)
class AdAc:
    layers = LayerAcc()
    obs = MetaAcc("obs")
    var = MetaAcc("var")
    obsm = MultiAcc("obsm")
    varm = MultiAcc("varm")
    obsp = GraphAcc("obsp")
    varp = GraphAcc("varp")

    def __getitem__(self, i: IdxStr2D) -> AdPath:
        def get(ad: AnnData) -> pd.api.extensions.ExtensionArray | NDArray[Any]:
            return np.asarray(ad[i].X)  # TODO: pandas

        return AdPath(f"A[{i[0]!r}, {i[1]!r}]", get)

    @classmethod
    def resolve(cls, spec: str) -> AdPath:
        inst = cls()
        acc, *pieces = spec.split(".")
        pieces = list(pieces)
        if not hasattr(inst, acc):
            return inst[tuple(int(acc), int(pieces[0]))]
        current = getattr(inst, acc)
        if isinstance(current, (GraphAcc, LayerAcc, MetaAcc, MultiAcc)):
            current = current[pieces.pop(0)]
        if isinstance(current, MultiVecAcc):
            current = current[int(pieces.pop(0))]
        elif isinstance(current, (LayerVecAcc, GraphVecAcc)):
            current = current[int(pieces.pop(0)), int(pieces.pop(0))]
        return current
