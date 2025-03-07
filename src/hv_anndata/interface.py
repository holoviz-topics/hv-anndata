"""Anndata interface for holoviews."""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Any, TypedDict, cast

import holoviews as hv
from anndata import AnnData

from .accessors import AdAc, AdPath

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    class Dims(TypedDict):
        """Holoviews Dimensions."""

        kdims: list[str] | None
        vdims: list[str] | None


ACCESSOR = AdAc()


class _Raise(Enum):
    Sentry = auto()


class AnnDataInterface(hv.core.Interface):
    """Anndata interface for holoviews."""

    types = (AnnData,)
    datatype = "anndata"

    @classmethod
    def init(
        cls,
        eltype: hv.Element,  # noqa: ARG003
        data: AnnData,
        kdims: list[str] | None,
        vdims: list[str] | None,
    ) -> tuple[AnnData, Dims, dict[str, Any]]:
        """Initialize the interface."""
        return data, {"kdims": kdims, "vdims": vdims}, {}

    @classmethod
    def validate(cls, dataset: hv.Dataset, vdims=True):
        dims = "all" if vdims else "key"
        not_found = [
            d
            for d in cast(list[AdPath], dataset.dimensions(dims, label=False))
            if not d.isin(dataset.data)
        ]
        if not_found:
            msg = (
                "Supplied data does not contain specified "
                "dimensions, the following dimensions were "
                f"not found: {not_found!r}"
            )
            raise hv.DataError(msg, cls)

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
    ) -> np.ndarray | pd.api.extensions.ExtensionArray:
        """Retrieve values for a dimension."""
        dim = cast(AdPath, data.get_dimension(dim))
        adata = cast(AnnData, data.data)
        return dim(adata)

    @classmethod
    def dimension_type(
        cls, data: hv.Dataset, dim: hv.Dimension | str
    ) -> np.dtype | pd.api.extensions.ExtensionDtype:
        """Get the data type for a dimension."""
        dim = cast(AdPath, data.get_dimension(dim))
        adata = cast(AnnData, data.data)
        return dim(adata).dtype


def register() -> None:
    """Register the data type and interface with holoviews."""
    if AnnDataInterface.datatype not in hv.core.data.datatypes:
        hv.core.data.datatypes.append(AnnDataInterface.datatype)
    hv.core.Interface.register(AnnDataInterface)


def unregister() -> None:
    """Unregister the data type and interface with holoviews."""
    hv.core.data.datatypes.remove(AnnDataInterface.datatype)
    del hv.core.Interface.interfaces[AnnDataInterface.datatype]
