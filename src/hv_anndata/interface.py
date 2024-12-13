"""Anndata interface for holoviews."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, TypedDict, cast, overload

import holoviews as hv
from anndata import AnnData

if TYPE_CHECKING:
    import numpy as np

    class Dims(TypedDict):
        """Holoviews Dimensions."""

        kdims: list[str] | None
        vdims: list[str] | None


class _Raise(Enum):
    Sentry = auto()


@dataclass
class _AnnDataProxy:
    adata: AnnData

    @overload
    def get(self, k: str, /, default: None = None) -> np.ndarray | None: ...
    @overload
    def get(self, k: str, /, default: np.ndarray | _Raise) -> np.ndarray: ...
    def get(
        self, k: str, /, default: np.ndarray | _Raise | None = None
    ) -> np.ndarray | None:
        k_orig = k
        if "." not in k:
            if default is not _Raise.Sentry and k not in self.adata.var_names:
                return default
            return self.adata[:, k].X.flatten()
        attr_name, k = k.split(".", 1)
        attr = getattr(self.adata, attr_name)
        if "." not in k:
            if default is not _Raise.Sentry and k not in attr:
                return default
            return attr[k]
        k, i = k.split(".", 1)
        arr = attr[k]
        if "." not in i:
            if default is not _Raise.Sentry and not (0 <= int(i) < arr.shape[1]):
                return default
            return arr[:, int(i)]
        raise KeyError(k_orig)

    def __contains__(self, k: str) -> bool:
        return self.get(k) is not None

    def __getitem__(self, k: str) -> object:
        return self.get(k, _Raise.Sentry)

    def __len__(self) -> int:
        return len(self.adata)


class AnnDataInterface(hv.core.Interface):
    """Anndata interface for holoviews."""

    types = (AnnData,)
    datatype = "anndata"

    @classmethod
    def init(
        cls,
        eltype: hv.Element,  # noqa: ARG003
        data: AnnData | _AnnDataProxy,
        kdims: list[str] | None,
        vdims: list[str] | None,
    ) -> tuple[_AnnDataProxy, Dims, dict[str, Any]]:
        """Initialize the interface."""
        proxy = _AnnDataProxy(data) if isinstance(data, AnnData) else data
        return proxy, {"kdims": kdims, "vdims": vdims}, {}

    @classmethod
    def values(
        cls,
        data: hv.Dataset,
        dim: hv.Dimension | str,
        expanded: bool = True,  # noqa: FBT001, FBT002, ARG003
        flat: bool = True,  # noqa: FBT001, FBT002, ARG003
        *,
        compute: bool = True,  # noqa: ARG003
        keep_index: bool = False,  # noqa: ARG003
    ) -> np.ndarray:
        """Retrieve values for a dimension."""
        dim = data.get_dimension(dim)
        proxy = cast(_AnnDataProxy, data.data)
        return proxy[dim.name]

    @classmethod
    def dimension_type(cls, data: hv.Dataset, dim: hv.Dimension | str) -> np.dtype:
        """Get the data type for a dimension."""
        dim = data.get_dimension(dim)
        proxy = cast(_AnnDataProxy, data.data)
        return proxy[dim.name].dtype


def register() -> None:
    """Register the data type and interface with holoviews."""
    if AnnDataInterface.datatype not in hv.core.data.datatypes:
        hv.core.data.datatypes.append(AnnDataInterface.datatype)
    hv.core.Interface.register(AnnDataInterface)
