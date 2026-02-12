"""Anndata interface for holoviews."""

from __future__ import annotations

import warnings
from itertools import product
from typing import TYPE_CHECKING, TypedDict, cast

import holoviews as hv
import numpy as np
import pandas as pd
from anndata import AnnData
from anndata.acc import AdAcc
from holoviews.core.data import Dataset
from holoviews.core.data.grid import GridInterface
from holoviews.core.data.interface import DataError
from holoviews.core.element import Element
from holoviews.core.ndmapping import NdMapping, item_check, sorted_context
from holoviews.core.util import (
    cartesian_product,
    expand_grid_coords,
    get_param_values,
    unique_iterator,
)
from holoviews.element.raster import SheetCoordinateSystem

from ._ref import AdDim

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from numbers import Number
    from typing import Any, Literal, TypeVar

    from holoviews.core.dimension import Dimension
    from numpy.typing import NDArray

    # https://github.com/holoviz/holoviews/blob/5653e2804f1ab44a8f655a5fea6fa5842e234120/holoviews/core/data/__init__.py#L594-L607
    SelectionValues = (
        tuple[Number, Number]
        | Sequence[Number]
        | Callable[[NDArray], NDArray[np.bool_]]
    )
    # https://github.com/holoviz/holoviews/blob/5653e2804f1ab44a8f655a5fea6fa5842e234120/holoviews/core/data/__init__.py#L624-L627
    SelectionSpec = type | Callable | str

    T = TypeVar("T")
    ValueType = np.ndarray | pd.api.extensions.ExtensionArray


MSG_1D = "AnnData Dataset key dimensions must map onto one axis: obs or var."


class Dims(TypedDict):
    """Holoviews Dimensions."""

    kdims: Sequence[Dimension] | None
    vdims: Sequence[Dimension] | None


class AnnDataInterface(hv.core.Interface):
    """Anndata interface for holoviews."""

    types = (AnnData,)
    datatype = "anndata"

    @classmethod
    def init(
        cls,
        eltype: hv.Element,  # noqa: ARG003
        data: AnnData,
        kdims: list[Dimension] | None,
        vdims: list[Dimension] | None,
    ) -> tuple[AnnData, Dims, dict[str, Any]]:
        """Initialize the interface."""
        if not isinstance(data, AnnData):
            msg = f"{cls.__name__} only supports AnnData."
            raise TypeError(msg)
        key_dimensions = [AdAcc.from_dimension(d) for d in kdims or []]
        value_dimensions = [AdAcc.from_dimension(d) for d in vdims or []]
        vdim = value_dimensions[0] if value_dimensions else None
        ndims = len(vdim.axes) if vdim else 1
        if not cls.gridded and ndims > 1:
            msg = f"{cls.__name__} cannot handle gridded data."
            raise ValueError(msg)
        if cls.gridded and ndims == 1:
            msg = f"{cls.__name__} cannot handle tabular data."
            raise ValueError(msg)
        return data, {"kdims": key_dimensions, "vdims": value_dimensions}, {}

    @classmethod
    def axes(cls, dataset: Dataset) -> tuple[Literal["obs", "var"], ...]:
        """Detect if the data is gridded or columnar and along which axes it is indexed."""  # noqa: E501
        vdim = cls._dim(dataset, dataset.vdims[0]) if dataset.vdims else None
        if len(dataset.kdims) == 0 and vdim:
            # e.g. Violin plot doesn’t have kdims
            return tuple(vdim.axes)

        ndims = len(vdim.axes) if vdim else 1
        if ndims not in (allowed := {1, len(dataset.kdims)}):
            msg = (
                "AnnData Dataset with multi-dimensional data must declare "
                f"corresponding key dimensions ({ndims} ∉ {allowed})."
            )
            raise DataError(msg)

        # Check that all key dimensions map onto the same axis
        kdims = [cls._dim(dataset, k) for k in dataset.dimensions()]
        if ndims != 1:
            kdims = kdims[:2]
        if any(len(dim.axes) > 1 for dim in kdims):
            raise DataError(MSG_1D)

        # Determine spanned axes from key dimensions (order matters)
        axes: tuple[Literal["obs", "var"], ...] = tuple({
            next(iter(dim.axes)): None for dim in kdims
        })

        # If vdim is ("obs", "obs") | ("var", "var") use it directly
        if vdim and vdim.axes in {("obs", "obs"), ("var", "var")}:
            if next(iter(vdim.axes)) not in axes:
                msg = "AnnData Dataset in gridded mode vdim axes must match kdims."
                raise DataError(msg)
            axes = tuple(vdim.axes)
        # Check if vdim axes match key dimension axes
        elif ndims != len(axes):
            msg = (
                "AnnData Dataset in tabular mode must reference data along either the "
                "obs or the var axis, not both."
            )
            raise DataError(msg)

        return axes

    @staticmethod
    def _dim(dataset: Dataset, k: Dimension | str | int) -> AdDim:
        if isinstance(k, int):
            k = cast("Dimension", dataset.get_dimension(k, strict=True))
        return AdAcc.from_dimension(
            (dataset.get_dimension(k) or AdAcc.resolve(k)) if isinstance(k, str) else k
        )

    @classmethod
    def validate(cls, dataset: Dataset, vdims: bool = True) -> None:  # noqa: FBT001, FBT002
        """Check if all dimensions (or key dimensions if `vdims==False`) are present."""
        dims = "all" if vdims else "key"
        not_found = [
            d
            for d in cast("list[Dimension]", dataset.dimensions(dims, label=False))
            if isinstance(d, AdDim) and not d.isin(dataset.data)
        ]
        if not_found:
            msg = (
                "Supplied data does not contain specified "
                "dimensions, the following dimensions were "
                f"not found: {not_found!r}"
            )
            raise DataError(msg, cls)
        axes = cls.axes(dataset)
        del axes

    @classmethod
    def validate_selection_dim(cls, dim: AdDim, action: str) -> Literal["obs", "var"]:
        """Validate dimension as valid axis to select on."""
        if len(dim.axes) > 1:
            raise DataError(MSG_1D)
        [ax] = dim.axes
        # TODO: support ranges and sequences  # noqa: TD003
        if ax not in {"obs", "var"}:  # pragma: no cover
            msg = f"Cannot {action} along unknown axis: {ax}"
            raise AssertionError(msg)
        return ax

    @classmethod
    def selection_masks(
        cls, dataset: Dataset, selection: Mapping[Dimension | str, SelectionValues]
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Generate boolean masks along obs and var axes."""
        adata = cast("AnnData", dataset.data)
        obs = var = None
        for k, v in selection.items():
            dim = cls._dim(dataset, k)
            ax = cls.validate_selection_dim(dim, "select")
            values = dim(adata)
            mask = None
            sel = slice(*v) if isinstance(v, tuple) else v
            if isinstance(sel, slice):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", r"invalid value encountered")
                    if sel.start is not None:
                        mask = sel.start <= values
                    if sel.stop is not None:
                        stop_mask = values < sel.stop
                        mask = stop_mask if mask is None else (mask & stop_mask)
            elif isinstance(sel, set | list):
                iter_slcs = []
                for ik in sel:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", r"invalid value encountered")
                        iter_slcs.append(values == ik)
                mask = np.logical_or.reduce(iter_slcs)
            elif callable(sel):
                mask = sel(values)
            else:
                mask = values == sel
            if ax == "obs":
                obs = mask if obs is None else (obs & mask)
            elif ax == "var":
                var = mask if var is None else (var & mask)
        return obs, var

    @classmethod
    def select(
        cls,
        dataset: Dataset,
        *,
        selection_mask: Sequence[SelectionSpec] | None = None,
        **selection: SelectionValues,
    ) -> AnnData:
        """Select along obs and var axes."""
        if selection_mask is None:
            obs, var = cls.selection_masks(dataset, selection)
        else:
            axes = cls.axes(dataset)
            obs, var = (
                (selection_mask, None) if axes == ("obs",) else (None, selection_mask)
            )
        adata = cast("AnnData", dataset.data)
        if obs is None:
            return adata if var is None else dataset.data[:, var]
        if var is None:
            return adata[obs]
        return adata[obs, var]

    @classmethod
    def reindex(
        cls,
        dataset: Dataset,
        kdims: list[Dimension] | None = None,  # noqa: ARG003
        vdims: list[Dimension] | None = None,  # noqa: ARG003
    ) -> AnnData:
        """Reindex the data (a no-op)."""
        return dataset.data

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
    ) -> ValueType:
        """Retrieve values for a dimension."""
        dim = cls._dim(data, dim)
        adata = cast("AnnData", data.data)
        values = dim(adata)
        if not keep_index and isinstance(values, pd.Series):
            values = values.values
        elif flat and values.ndim > 1:
            assert not isinstance(values, pd.api.extensions.ExtensionArray)  # noqa: S101
            values = values.flatten()
        return values

    @classmethod
    def dtype(
        cls, data: Dataset, dim: Dimension | str | int
    ) -> np.dtype | pd.api.extensions.ExtensionDtype:
        """Get the data type for a dimension."""
        dim = cls._dim(data, dim)
        adata = cast("AnnData", data.data)
        return dim(adata).dtype

    @classmethod
    def dimension_type(cls, dataset: Dataset, dim: Dimension | str) -> type:
        """Get the scalar type for a dimension (e.g. `np.int64`)."""
        return cls.dtype(dataset, dim).type

    @classmethod
    def iloc(
        cls, dataset: Dataset, index: tuple[slice[int | None], slice[None]]
    ) -> AnnData:
        """Implement `Dataset.iloc`."""
        rows, cols = index
        axes = cls.axes(dataset)
        adata = cast("AnnData", dataset.data)

        if (idx := cls._iloc_2d(axes, rows, cols)) is not None:
            return adata[idx]

        match axes[0]:
            case "var":
                return adata[:, rows]
            case "obs":
                return adata[rows]

    @classmethod
    def _iloc_2d(
        cls,
        axes: tuple[Literal["obs", "var"], ...],
        rows: slice[int | None],  # noqa: ARG003
        cols: slice[None],
    ) -> tuple[slice[int | None], slice[int | None]] | None:
        """Validate indexing. Overridden in `AnnDataGriddedInterface`."""
        if cols != slice(None):
            msg = (
                f"When indexing using .iloc on {axes[0]} indexed data, "
                "you may only select rows along that dimension, "
                "i.e. you may not provide a column selection."
            )
            raise IndexError(msg)
        return None

    @classmethod
    def aggregate(
        cls,
        dataset: Dataset,
        kdims: list[Dimension | str],
        function: Callable[ValueType, ValueType],
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[pd.DataFrame, list[Dimension]]:
        """Aggregate the current view."""
        agg = Dataset(
            dataset.dframe(),
            kdims=kdims,
            vdims=[vd for vd in dataset.vdims if vd not in kdims],
        )
        return agg.interface.aggregate(agg, kdims, function=function, **kwargs)

    @classmethod
    def unpack_scalar(
        cls,
        dataset: Dataset,  # noqa: ARG003
        data: AnnData | pd.DataFrame,
    ) -> Any:  # noqa: ANN401
        """Unpacks scalar data if it is a DataFrame containing a single value."""
        if isinstance(data, AnnData):
            return data
        if len(data) != 1 or len(data.columns) > 1:
            return data
        return data.iloc[0, 0]

    @classmethod
    def groupby(
        cls,
        dataset: Dataset,
        dimensions: Sequence[str | AdDim],
        container_type: type[T],
        group_type: type[Dataset | dict] | Literal["raw"],
        **kwargs: Any,  # noqa: ANN401
    ) -> T:
        """Group the dataset along the provided dimensions."""
        values = {}
        adata = cast("AnnData", dataset.data)
        for k in dimensions:
            dim = cls._dim(dataset, k)
            cls.validate_selection_dim(dim, "group")
            values[k] = unique_iterator(dim(adata))

        # Get dimensions information
        dimensions = [dataset.get_dimension(d) for d in dimensions]
        kdims = [kdim for kdim in dataset.kdims if kdim not in dimensions]

        # Update the kwargs appropriately for Element group types
        group_kwargs = {}
        group_type = dict if group_type == "raw" else group_type
        if issubclass(group_type, Element):
            group_kwargs.update(get_param_values(dataset))
            group_kwargs["kdims"] = kdims
        group_kwargs.update(kwargs)

        # Generate the groupings and then start iterating over them
        selectors = (
            dict(zip(values, combo, strict=False))
            for combo in product(*values.values())
        )
        groups = []
        for sel in selectors:
            group = dataset.select(sel)
            if group_type == "raw":
                group = group.data
            if type(group) is not group_type:
                group = group.clone(new_type=group_type)
            groups.append((tuple(sel.values()), group))
        if issubclass(container_type, NdMapping):
            with item_check(enabled=False), sorted_context(enabled=False):
                return container_type(groups, kdims=dimensions)
        else:
            return container_type(groups)


class AnnDataGriddedInterface(AnnDataInterface):
    """Anndata interface for holoviews."""

    datatype = "anndata-gridded"
    gridded = True

    @classmethod
    def shape(cls, dataset: Dataset, *, gridded: bool = False) -> tuple[int, int]:
        """Retrieve shape of 2D data."""
        del gridded
        ax1, ax2 = cls.axes(dataset)
        return len(getattr(dataset.data, ax1)), len(getattr(dataset.data, ax2))

    @classmethod
    def _iloc_2d(
        cls,
        axes: tuple[Literal["obs", "var"], ...],
        rows: slice[int | None],
        cols: slice[None],
    ) -> tuple[slice[int | None], slice[int | None]] | None:
        if axes[0] != axes[1]:
            if axes[0] == "var":
                return (cols, rows)
            return (rows, cols)
        if cols != slice(None):
            ax = axes[0]
            msg = (
                f"When indexing using .iloc on pairwise variables "
                f"(in this case {ax}p) you may only index on rows, "
                f"i.e. index using `dataset.iloc[{ax}s]`, "
                f"not along two axes, as in `dataset[{ax}s, {ax}s2]).`"
            )
            raise IndexError(msg)
        return None

    @classmethod
    def coords(
        cls,
        dataset: Dataset,
        dim: Dimension | str,
        ordered: bool = False,  # noqa: FBT001, FBT002
        *,
        expanded: bool = False,
        edges: bool = False,
    ) -> NDArray[np.float64]:
        """Get the coordinates along a dimension.

        Ordered ensures coordinates are in ascending order and expanded creates
        ND-array matching the dimensionality of the dataset.
        """  # noqa: DOC201
        dim = cls._dim(dataset, dim)
        vdim = dataset.vdims[0]
        if expanded:
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
            # TODO: don’t do this, it’s broken if diff(shape) == 1  # noqa: TD003
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
    ) -> ValueType:
        """Retrieve values for a dimension."""
        dim = cls._dim(data, dim)
        adata = cast("AnnData", data.data)
        if dim in data.kdims and isinstance(data, SheetCoordinateSystem):
            # On 2D datasets we generate synthetic coordinates
            [ax] = dim.axes
            return np.arange(len(getattr(adata, ax)))

        # Axes are transposed relative to the key-dimension order.
        # When flattened, this transpose is omitted so arrays follow
        # hierarchical (row-major) ordering.
        transpose = [d.axes for d in data.kdims] == [{"obs"}, {"var"}]
        match len(dim.axes), expanded:
            case 1, True:
                obs, var = cls._expand_grid(data)
                idx = var if dim.axes == {"var"} else obs
                coords = dim(adata)
                values = coords[idx]
                if transpose != flat:
                    values = values.T
            case 1, False:
                values = dim(adata)
            case _, True:
                values = dim(adata)
                if transpose != flat:
                    values = values.T
            case _:
                assert dim in data.vdims, "test_init_errors[tab-x_2d] prevents 2D kdims"  # noqa: S101
                error = (
                    "When requesting data for a value dimension, "
                    "it is invalid to request expanded=False. "
                    "Value dimension arrays must cover the whole "
                    "multi-dimensional space."
                )
                raise ValueError(error)

        if not keep_index and isinstance(values, pd.Series):
            values = values.values
        if flat and values.ndim > 1:
            if isinstance(values, pd.api.extensions.ExtensionArray):
                values = values.to_numpy()
            values = values.flatten()
        return values

    @classmethod
    def _expand_grid(cls, data: Dataset) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        obs, var = cartesian_product(
            [np.arange(len(data.data.obs)), np.arange(len(data.data.var))],
            flat=False,
        )
        return obs, var

    @classmethod
    def irregular(cls, dataset: Dataset, dim: Dimension | str) -> Literal[False]:
        """Tell whether a dimension is irregular (i.e. has multi-dimensional coords)."""
        del dim
        del dataset
        return False


def register() -> None:
    """Register the data type and interface with holoviews."""
    if TYPE_CHECKING:
        assert isinstance(hv.element.Image.datatype, list)
    if AnnDataInterface.datatype not in hv.core.data.datatypes:
        hv.core.data.datatypes.append(AnnDataInterface.datatype)
    if AnnDataGriddedInterface.datatype not in hv.core.data.datatypes:
        hv.core.data.datatypes.append(AnnDataGriddedInterface.datatype)
    if AnnDataGriddedInterface.datatype not in hv.element.Image.datatype:
        hv.element.Image.datatype.append(AnnDataGriddedInterface.datatype)
    hv.core.Interface.register(AnnDataInterface)
    hv.core.Interface.register(AnnDataGriddedInterface)


def unregister() -> None:  # pragma: no cover
    """Unregister the data type and interface with holoviews."""
    if TYPE_CHECKING:
        assert isinstance(hv.element.Image.datatype, list)
    hv.core.data.datatypes.remove(AnnDataInterface.datatype)
    hv.core.data.datatypes.remove(AnnDataGriddedInterface.datatype)
    hv.element.Image.datatype.remove(AnnDataGriddedInterface.datatype)
    del hv.core.Interface.interfaces[AnnDataInterface.datatype]
    del hv.core.Interface.interfaces[AnnDataGriddedInterface.datatype]
