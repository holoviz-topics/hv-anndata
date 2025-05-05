"""Anndata interface for holoviews."""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

import holoviews as hv
import numpy as np
import pandas as pd
from anndata import AnnData
from holoviews.core.data import Dataset
from holoviews.core.data.grid import GridInterface
from holoviews.core.data.interface import DataError
from holoviews.core.util import expand_grid_coords
from holoviews.element.raster import SheetCoordinateSystem

from .accessors import AdAc, AdPath

if TYPE_CHECKING:
    from holoviews.core.dimension import Dimension

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
        key_dimensions = []
        for d in kdims or []:
            if not isinstance(d, AdPath):
                label = d.label if d.name != d.label else None
                d = AdAc.resolve(d.name)
                if label is not None:
                    d.label = label
            key_dimensions.append(d)

        value_dimensions = []
        for d in vdims or []:
            if not isinstance(d, AdPath):
                label = d.label if d.name != d.label else None
                d = AdAc.resolve(d.name)
                if label is not None:
                    d.label = label
            value_dimensions.append(d)
        vdim = value_dimensions and value_dimensions[0]
        ndim = 1 if not vdim else vdim(data).ndim
        if not cls.gridded and ndim > 1:
            msg = "AnnDataInterface cannot handle gridded data."
            raise ValueError(msg)
        if cls.gridded and ndim == 1:
            msg = "AnnDataGriddedInterface cannot handle tabular data."
            raise ValueError(msg)
        return data, {"kdims": key_dimensions, "vdims": value_dimensions}, {}

    @classmethod
    def axes(cls, dataset: Dataset) -> tuple[str, ...]:
        """Detects if the data is gridded or columnar and along which axes it is indexed."""
        dims = dataset.dimensions()
        vdim = dataset.vdims and dataset.vdims[0]
        ndim = 1 if not vdim else vdim(dataset.data).ndim
        axes, shapes = [], []
        if ndim > 1:
            # Gridded data case, ensure that the key dimensions (i.e. the 2D indexes)
            # map onto the obs and var axes.
            if len(dataset.kdims) != ndim:
                msg = (
                    "AnnData Dataset with multi-dimensional data must declare "
                    "corresponding key dimensions."
                )
                raise DataError(msg)
            for dim in dims[:2]:
                label = dim.name.lstrip("A.").replace("['", ".").replace("']", "")
                if label.startswith("obs"):
                    axes.append("obs")
                elif label.startswith("var"):
                    axes.append("var")
                else:
                    msg = (
                        "AnnData Dataset key dimensions must map onto either obs or var axes. "
                        "Cannot use multi-dimensional array as index."
                    )
                    raise DataError(msg)
                dim_shape = dim(dataset.data).shape
                if len(dim_shape) > 1:
                    msg = (
                        "AnnData Dataset key dimensions must map onto either obs or var axes. "
                        "Cannot use multi-dimensional array as index."
                    )
                    raise DataError(msg)
                shapes.append(dim_shape)
            return tuple(axes)

        # Tabular case where all dimensions must map onto either the obs or var dimension.
        for dim in dims:
            dim_shape = dim(dataset.data).shape
            if len(dim_shape) > 1:
                msg = (
                    "AnnData Dataset with multi-dimensional data must declare "
                    "corresponding key dimensions."
                )
                raise DataError(msg)
            label = (
                repr(dim)
                .lstrip("A.")
                .replace("['", ".")
                .replace("']", "")
                .replace(" ", "")
            )
            if label.startswith("obs"):
                axes.append("obs")
            elif label.startswith("var"):
                axes.append("var")
            elif label.startswith("layers") or dim.name.startswith("A["):
                # Detect if dimension was sliced along obs or var dimension
                if "[:" in label:
                    axes.append("obs")
                elif ":]" in label:
                    axes.append("var")
        if len(set(axes)) != 1:
            msg = (
                "AnnData Dataset in tabular mode must reference data along either the "
                "obs or the var axis, not both."
            )
            raise DataError(msg)
        return (axes[0],)

    @classmethod
    def validate(cls, dataset: Dataset, *, vdims: bool = True) -> None:
        dims = "all" if vdims else "key"
        not_found = [
            d
            for d in cast("list[AdPath]", dataset.dimensions(dims, label=False))
            if isinstance(d, AdPath) and not d.isin(dataset.data)
        ]
        if not_found:
            msg = (
                "Supplied data does not contain specified "
                "dimensions, the following dimensions were "
                f"not found: {not_found!r}"
            )
            raise DataError(msg, cls)
        axes = cls.axes(dataset)

    @classmethod
    def select(cls, dataset: Dataset, selection_mask=None, **selection):
        obs_selections, var_selections = {}, {}
        for k, v in selection.items():
            if k.startswith("obs."):
                obs_selections[k[4:]] = v
            elif k.startswith("A.obs['"):
                obs_selections[k[7:].replace("']", "")] = v
            elif k.startswith("var."):
                var_selections[k[4:]] = v
            elif k.startswith("A.var['"):
                var_selections[k[7:].replace("']", "")] = v
        if obs_selections:
            obs = Dataset(dataset.data.obs).select(**obs_selections).data.index
        else:
            obs = slice(None)
        if var_selections:
            var = Dataset(dataset.data.var).select(**var_selections).data.index
        else:
            var = slice(None)
        return dataset.data[obs, var]

    @classmethod
    def values(
        cls,
        data: Dataset,
        dim: Dimension | str,
        expanded: bool = True,  # noqa: FBT001, FBT002, ARG003
        flat: bool = True,  # noqa: FBT001, FBT002
        *,
        compute: bool = True,  # noqa: ARG003
        keep_index: bool = False,
    ) -> np.ndarray | pd.api.extensions.ExtensionArray:
        """Retrieve values for a dimension."""
        dim = cast("AdPath", data.get_dimension(dim))
        adata = cast("AnnData", data.data)
        values = dim(adata)
        if not keep_index and isinstance(values, pd.Series):
            values = values.array
        elif flat and values.ndim > 1:
            assert not isinstance(values, pd.api.extensions.ExtensionArray)  # noqa: S101
            values = values.flatten()
        return values

    @classmethod
    def dtype(
        cls, data: Dataset, dim: Dimension | str
    ) -> np.dtype | pd.api.extensions.ExtensionDtype:
        """Get the data type for a dimension."""
        dim = cast("AdPath", data.get_dimension(dim))
        adata = cast("AnnData", data.data)
        return dim(adata).dtype

    @classmethod
    def dimension_type(cls, dataset: Dataset, dim):
        return cls.dtype(dataset, dim).type

    @classmethod
    def iloc(cls, dataset: Dataset, index):
        rows, cols = index
        axes = cls.axes(dataset)
        if cols != slice(None):
            msg = (
                f"When indexing using .iloc on {axes[0]} indexed data you may only select "
                "rows along that dimension, i.e. you may not provide a column selection. "
            )
            raise IndexError(msg)
        if axes[0] == "var":
            return dataset.data[:, rows]
        if axes[0] == "obs":
            return dataset.data[rows]


class AnnDataGriddedInterface(AnnDataInterface):
    """Anndata interface for holoviews."""

    datatype = "anndata-gridded"
    gridded = True

    @classmethod
    def shape(cls, dataset: Dataset, *, gridded: bool = False) -> tuple[int, int]:
        del gridded
        ax1, ax2 = cls.axes(dataset)
        return (len(getattr(dataset.data, ax1)), len(getattr(dataset.data, ax2)))

    @classmethod
    def iloc(cls, dataset: Dataset, index):
        rows, cols = index
        ax1, ax2 = cls.axes(dataset)
        if ax1 != ax2:
            if ax1 == "var":
                return dataset.data[cols, rows]
            return dataset.data[rows, cols]

        if cols != slice(None):
            msg = (
                f"When indexing using .iloc on pairwise variables "
                f"(in this case {ax1}p) you may only index on rows, "
                f"i.e. index using `dataset.iloc[{ax1}s]`, "
                f"not along two axes, as in `dataset[{ax1}s, {ax1}s2]).`"
            )
            raise IndexError(msg)
        if ax1 == "var":
            return dataset.data[:, rows]
        if ax2 == "obs":
            return dataset.data[rows]
        return None

    @classmethod
    def coords(
        cls,
        dataset: Dataset,
        dim: Dimension,
        *,
        ordered: bool = False,
        expanded: bool = False,
        edges: bool = False,
    ):
        """Get the coordinates along a dimension.

        Ordered ensures coordinates are in ascending order and expanded creates
        ND-array matching the dimensionality of the dataset.
        """
        dim = dataset.get_dimension(dim, strict=True)
        irregular = cls.irregular(dataset, dim)
        vdim = dataset.vdims[0]
        if irregular or expanded:
            data = expand_grid_coords(dataset, dim)
            if edges and data.shape == vdim(dataset.data).shape:
                data = GridInterface._infer_interval_breaks(data, axis=1)  # noqa: SLF001
                data = GridInterface._infer_interval_breaks(data, axis=0)  # noqa: SLF001
            return data

        data = dim(dataset.data)
        if ordered and np.all(data[1:] < data[:-1]):
            data = data[::-1]
        shape = cls.shape(dataset, gridded=True)
        if dim in dataset.kdims:
            idx = dataset.get_dimension_index(dim)
            is_edges = (
                dim in dataset.kdims
                and len(shape) == dataset.ndims
                and len(data) == (shape[dataset.ndims - idx - 1] + 1)
            )
        else:
            is_edges = False
        if edges and not is_edges:
            data = GridInterface._infer_interval_breaks(data)  # noqa: SLF001
        elif not edges and is_edges:
            data = data[:-1] + np.diff(data) / 2.0
        return data

    @classmethod
    def values(
        cls,
        data: Dataset,
        dim: Dimension | str,
        expanded: bool = True,  # noqa: FBT001, FBT002
        flat: bool = True,  # noqa: FBT001, FBT002
        *,
        compute: bool = True,  # noqa: ARG003
        keep_index: bool = False,
    ) -> np.ndarray | pd.api.extensions.ExtensionArray:
        """Retrieve values for a dimension."""
        dim = cast("AdPath", data.get_dimension(dim))
        idx = data.get_dimension_index(dim)
        adata = cast("AnnData", data.data)
        axes = cls.axes(data)
        if idx < 2 and isinstance(data, SheetCoordinateSystem):
            # On 2D datasets we generate synthetic coordinates
            ax = axes[idx]
            return np.arange(len(getattr(adata, ax)))
        if expanded and dim in data.kdims:
            values = expand_grid_coords(data, dim)
        else:
            values = dim(adata)
        if not keep_index and isinstance(values, pd.Series):
            values = values.array
        elif flat and values.ndim > 1:
            assert not isinstance(values, pd.api.extensions.ExtensionArray)  # noqa: S101
            values = values.flatten()
        return values

    @classmethod
    def irregular(cls, dataset: Dataset, dim: Dimension | str) -> Literal[False]:
        """Tell whether a dimension is irregular (i.e. has multi-dimensional coords)."""
        del dim
        del dataset
        return False


def register() -> None:
    """Register the data type and interface with holoviews."""
    if AnnDataInterface.datatype not in hv.core.data.datatypes:
        hv.core.data.datatypes.append(AnnDataInterface.datatype)
    if AnnDataGriddedInterface.datatype not in hv.core.data.datatypes:
        hv.core.data.datatypes.append(AnnDataGriddedInterface.datatype)
    if AnnDataGriddedInterface.datatype not in hv.element.Image.datatype:
        hv.element.Image.datatype.append(AnnDataGriddedInterface.datatype)
    hv.core.Interface.register(AnnDataInterface)
    hv.core.Interface.register(AnnDataGriddedInterface)


def unregister() -> None:
    """Unregister the data type and interface with holoviews."""
    hv.core.data.datatypes.remove(AnnDataInterface.datatype)
    hv.core.data.datatypes.remove(AnnDataGriddedInterface.datatype)
    del hv.core.Interface.interfaces[AnnDataInterface.datatype]
    del hv.core.Interface.interfaces[AnnDataGriddedInterface.datatype]
