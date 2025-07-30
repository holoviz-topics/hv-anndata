"""Interactive vizualization of AnnData dimension reductions with HoloViews and Panel."""  # noqa: E501

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

import anndata as ad
import bokeh
import bokeh.palettes
import colorcet as cc
import datashader as ds
import holoviews as hv
import holoviews.operation.datashader as hd
import numpy as np
import panel as pn
import panel_material_ui as pmui
import param
from bokeh.models.tools import BoxSelectTool, LassoSelectTool
from holoviews.operation import Operation
from panel.reactive import hold

if TYPE_CHECKING:
    from collections.abc import Sequence


DEFAULT_COLOR_BY = "cell_type"
CAT_CMAPS = {
    "Glasbey Cat10": cc.b_glasbey_category10,
    "Cat20": bokeh.palettes.Category20_20,
    "Glasbey cool": cc.glasbey_cool,
}
CONT_CMAPS = {
    "Viridis": bokeh.palettes.Viridis256,
    "Fire": cc.fire,
    "Blues": cc.blues,
}
DEFAULT_CAT_CMAP = cc.b_glasbey_category10
DEFAULT_CONT_CMAP = "viridis"


def _is_categorical(arr: np.ndarr) -> bool:
    return (
        arr.dtype.name in ["category", "categorical", "bool"]
        or np.issubdtype(arr.dtype, np.object_)
        or np.issubdtype(arr.dtype, np.str_)
    )


class labeller(Operation):
    """Add a Label element centered over categorical points."""

    column = param.String()

    max_labels = param.Integer(10)

    min_count = param.Integer(default=100)

    streams = param.List([hv.streams.RangeXY])

    x_range = param.Tuple(
        default=None,
        length=2,
        doc="""
       The x_range as a tuple of min and max x-value. Auto-ranges
       if set to None.""",
    )

    y_range = param.Tuple(
        default=None,
        length=2,
        doc="""
       The x_range as a tuple of min and max x-value. Auto-ranges
       if set to None.""",
    )

    def _process(self, el: hv.Dataset, key=None) -> hv.Labels:  # noqa: ARG002, ANN001
        if self.p.x_range and self.p.y_range:
            el = el[slice(*self.p.x_range), slice(*self.p.y_range)]

        df = el.dframe()
        xd, yd, cd = el.dimensions()[:3]
        col = self.p.column or cd.name
        result = (
            df.groupby(col)
            .agg(
                count=(col, "size"),  # count of rows per group
                x=(xd.name, "mean"),
                y=(yd.name, "mean"),
            )
            .query(f"count > {self.p.min_count}")
            .sort_values("count", ascending=False)
            .iloc[: self.p.max_labels]
            .reset_index()
        )
        return hv.Labels(result, ["x", "y"], col)


class ManifoldMapConfig(TypedDict, total=False):
    """Configuration options for manifold map plotting."""

    width: int
    """width of the plot (default: 300).
    If responsive is True, this is the minimum width."""
    height: int
    """minimum height of the plot (default: 300)
    If responsive is True, this is the minimum height."""
    datashading: bool
    """whether to apply datashader (default: True)"""
    show_labels: bool
    """whether to overlay labels at median positions (default: False)"""
    cmap: str | list[str]
    """colormap"""
    title: str
    """plot title (default: "")"""
    responsive: bool
    """whether to make the plot size-responsive. (default: True)"""


def create_manifoldmap_plot(
    x_data: np.ndarray,
    color_data: np.ndarray,
    x_dim: int,
    y_dim: int,
    color_by: str,
    xaxis_label: str,
    yaxis_label: str,
    *,
    categorical: bool | None = None,
    **config: Unpack[ManifoldMapConfig],
) -> hv.Element:
    """Create a comprehensive manifold map plot with options for datashading and labels.

    Parameters
    ----------
    x_data
        Array with shape n_obs by n_dimensions containing coordinates
    color_data
        Array with shape n_obs containing color values (categorical or continuous)
    x_dim
        Index to use for x-axis data
    y_dim
        Index to use for y-axis data
    color_by
        Name to give the coloring dimension
    xaxis_label
        Label for the x axis
    yaxis_label
        Label for the y axis
    categorical: bool or None, default=None
        Whether the data in color_by is categorical
    **config
        Additional configuration options including, see :class:`ManifoldMapConfig`.

    Returns
    -------
    HoloViews element with the configured plot

    """
    # Extract config with defaults
    width = config.get("width", 300)
    height = config.get("height", 300)
    datashading = config.get("datashading", True)
    show_labels = config.get("show_labels", False)
    cmap = config.get("cmap")
    title = config.get("title", "")
    responsive = config.get("responsive", True)

    # Determine if color data is categorical
    if categorical is None:
        categorical = _is_categorical(color_data)

    # Add a NaN category to handle and display data points with no category
    if categorical:
        color_data = np.where(
            color_data != color_data,
            "NaN",
            color_data,
        )  # np.nan != np.nan is True

    # Set colormap and plot options based on data type
    if categorical:
        n_unq_cat = len(np.unique(color_data))
        if cmap is None:
            cmap = DEFAULT_CAT_CMAP
        # Use subset of categorical colormap to preserve distinct colors
        cmap = cmap[:n_unq_cat]
        colorbar = False
        show_legend = True
    else:
        if cmap is None:
            cmap = DEFAULT_CONT_CMAP
        show_legend = False
        colorbar = True

    # Create basic plot
    dataset = hv.Dataset(
        (x_data[:, x_dim], x_data[:, y_dim], color_data),
        [xaxis_label, yaxis_label],
        color_by,
    )
    plot = dataset.to(hv.Points)

    # Options for standard (non-datashaded) plot
    plot_opts = dict(
        color=color_by,
        cmap=cmap,
        size=1,
        alpha=0.5,
        colorbar=colorbar,
        padding=0,
        tools=[
            "hover",
            BoxSelectTool(persistent=True),
            LassoSelectTool(persistent=True),
        ],
        show_legend=show_legend,
        legend_position="right",
    )

    # Apply different rendering based on configuration
    if not datashading:
        # Standard plot without datashading
        plot = plot.opts(**plot_opts)

    # Apply datashading with different approaches for categorical vs continuous
    elif categorical:
        plot = _apply_categorical_datashading(
            plot,
            color_data=color_data,
            color_by=color_by,
            cmap=cmap,
        )
    else:
        # For continuous data, take the mean
        aggregator = ds.mean(color_by)
        plot = hd.rasterize(plot, aggregator=aggregator)
        plot = hd.dynspread(plot, threshold=0.5)
        plot = plot.opts(
            cmap=cmap,
            colorbar=colorbar,
            tools=[
                "hover",
                BoxSelectTool(persistent=True),
                LassoSelectTool(persistent=True),
            ],
        )

    if categorical and show_labels:
        # Options for labels
        label_opts = dict(text_font_size="8pt", text_color="black")
        plot = plot * labeller(dataset).opts(**label_opts)

    if not responsive:
        plot = plot.opts(
            responsive=False,
            frame_height=height,
            frame_width=width,
        )
    else:
        plot = plot.opts(
            responsive=True,
            min_height=height,
            min_width=width,
        )

    # Apply final options to the plot
    return plot.opts(
        title=title,
        show_legend=show_legend,
    )


def _apply_categorical_datashading(
    plot: hv.Element,
    *,
    color_data: np.ndarray,
    color_by: str,
    cmap: Sequence[str],
) -> hv.Element:
    """Apply datashading to categorical data.

    Parameters
    ----------
    plot
        The base plot to apply datashading to
    color_data
        Category data for coloring
    color_by
        Name of the color variable
    cmap
        Colormap to use

    Returns
    -------
    Datashaded plot with a custom legend

    """
    # For categorical data, count by category
    aggregator = ds.count_cat(color_by)
    # Selector used as a workaround to display categorical counts per pixel
    # One day done directly in Bokeh, see https://github.com/bokeh/bokeh/issues/13354
    selector = ds.first(plot.kdims[0].name)
    plot = hd.rasterize(plot, aggregator=aggregator, selector=selector)
    plot = hd.dynspread(plot, threshold=0.5)
    unique_categories = np.unique(color_data)
    plot = plot.opts(
        cmap=cmap,
        tools=[
            "hover",
            BoxSelectTool(persistent=True),
            LassoSelectTool(persistent=True),
        ],
        # Override hover_tooltips to exclude the selector value
        hover_tooltips=list(unique_categories),
        # Don't include the selector heading
        selector_in_hovertool=False,
    )

    # Create a custom legend for datashaded categorical plot
    if len(unique_categories) > len(cmap):
        # cmap not long enough, cycle it
        cmap = cmap * (len(unique_categories) // len(cmap) + 1)
    color_key = dict(
        zip(unique_categories, cmap[: len(unique_categories)], strict=False)
    )
    legend_items = [
        hv.Points([0, 0], label=str(cat)).opts(color=color_key[cat], size=0)
        for cat in unique_categories
    ]
    legend = hv.NdOverlay(
        {
            str(cat): item
            for cat, item in zip(unique_categories, legend_items, strict=False)
        }
    ).opts(
        show_legend=True,
        legend_position="right",
        legend_limit=100,
        legend_cols=len(unique_categories) // 10 + 1,
    )
    return plot * legend


class ManifoldMap(pn.viewable.Viewer):
    """Interactive manifold map application for exploring AnnData objects.

    This application provides widgets to select dimensionality reduction methods,
    dimensions for x and y axes, coloring variables, and display options.

    Parameters
    ----------
    adata
        AnnData object to visualize
    reduction
        Initial dimension reduction method to use
    color_by_dim
        Color by dimension, one of 'obs' (default) or 'cols.
    color_by
        Initial variable to use for coloring
    colormap
        Initial colormap to use for coloring
    datashade
        Whether to enable datashading
    width
        Minimum width of the plot.
        If responsive is True, this is the minimum width.
    height
        Minimum height of the plot.
        If responsive is True, this is the minimum height.
    show_labels
        Whether to show labels
    show_widgets
        Whether to show control widgets
    responsive
        Whether to make the plot size-responsive

    """

    adata: ad.AnnData = param.ClassSelector(  # type: ignore[assignment]
        class_=ad.AnnData, doc="AnnData object to visualize"
    )
    reduction: str = param.Selector(  # type: ignore[assignment]
        doc="Dimension reduction method"
    )
    x_axis: str = param.Selector()  # type: ignore[assignment]
    y_axis: str = param.Selector()  # type: ignore[assignment]
    color_by_dim: str = param.Selector(  # type: ignore[assignment]
        default="obs",
        objects={"Observations": "obs", "Variables": "cols"},
        label="Color By",
    )
    color_by: str = param.Selector(  # type: ignore[assignment]
        doc="Coloring variable"
    )
    colormap: str = param.Selector()
    datashade: bool = param.Boolean(  # type: ignore[assignment]
        default=True,
        label="Datashader Rasterize For Large Datasets",
        doc="Whether to enable datashading",
    )
    var_reference: str | None = param.String(  # type: ignore[assignment]
        default=None,
        allow_None=True,
        doc="""
        Column name in .var to use for populating the variable names, default
        to the index names if not set.
    """,
    )
    width: int = param.Integer(default=300, doc="Minimum width of the plot")  # type: ignore[assignment]
    height: int = param.Integer(default=300, doc="Minimum height of the plot")  # type: ignore[assignment]
    show_labels: bool = param.Boolean(  # type: ignore[assignment]
        default=False,
        label="Overlay Labels For Categorical Coloring",
        doc="Whether to show labels",
    )
    show_widgets: bool = param.Boolean(  # type: ignore[assignment]
        default=True, doc="Whether to show control widgets"
    )
    responsive: bool = param.Boolean(  # type: ignore[assignment]
        default=True,
        doc="Whether to make the plot size-responsive",
    )
    _replot: bool = param.Event()  # type: ignore[assignment]

    def __init__(self, **params: object) -> None:
        """Initialize the ManifoldMapApp with the given parameters."""
        super().__init__(**params)
        self._categorical = False
        dr_options = list(self.adata.obsm.keys())
        self.param["reduction"].objects = dr_options
        if not self.reduction:
            self.reduction = dr_options[0]

        if self.var_reference:
            cols = list(self.adata.var[self.var_reference])
        else:
            cols = list(self.adata.var_names)
        self._color_options = {
            "obs": list(self.adata.obs.columns),
            "cols": cols,
        }
        copts = self._color_options[self.color_by_dim]
        self.param.color_by.objects = copts
        if not self.color_by:
            if (
                self.color_by_dim == "obs"
                and DEFAULT_COLOR_BY in self._color_options["obs"]
            ):
                self.color_by = DEFAULT_COLOR_BY
            else:
                self.color_by = self._color_options[self.color_by_dim][0]
        elif self.color_by not in copts:
            msg = f"color_by variable {self.color_by!r} not found."
            raise ValueError(msg)
        else:
            self._update_on_color_by()
        self._update_axes()

    @param.depends("color_by_dim", watch=True)
    def _on_color_by_dim(self) -> None:
        values = self._color_options[self.color_by_dim]
        self.param.color_by.objects = values
        self.color_by = values[0]

    @param.depends("color_by", watch=True)
    def _update_on_color_by(self) -> None:
        if not self.color_by:
            return
        old_is_categorical = self._categorical
        if self.color_by_dim == "obs":
            color_data = self.adata.obs[self.color_by].values
        elif self.color_by_dim == "cols":
            color_data = self.adata.obs_vector(self._get_var())
        self._categorical = _is_categorical(color_data)
        if old_is_categorical != self._categorical or not self.colormap:
            cmaps = CAT_CMAPS if self._categorical else CONT_CMAPS
            self.param.colormap.objects = cmaps
            self.colormap = next(iter(cmaps.values()))
        self._replot = True

    @hold()
    @param.depends("reduction", watch=True)
    def _update_axes(self) -> None:
        # Reset dimension options when reduction selection changes
        new_dims = self.get_dim_labels(self.reduction)
        vals = {
            "x_axis": new_dims[0],
            "y_axis": new_dims[1],
        }
        self.param.x_axis.objects = new_dims
        self.param.y_axis.objects = new_dims
        self.param.update(vals)

    def _get_var(self) -> str:
        if self.var_reference:
            var = self.adata.var.query(f'feature_name == "{self.color_by}"').index
            if len(var) > 1:
                msg = (
                    f"More than one vars found in {self.var_reference!r} "
                    f"for {self.color_by!r}."
                )
                raise RuntimeError(msg)
            var = var.item()
        else:
            var = self.color_by
        return var

    def get_reduction_label(self, dr_key: str) -> str:
        """Get a display label for a dimension reduction key.

        Parameters
        ----------
        dr_key
            The dimension reduction key

        Returns
        -------
        A formatted label for display

        """
        return dr_key.split("_")[1].upper() if "_" in dr_key else dr_key.upper()

    def get_dim_labels(self, dr_key: str) -> list[str]:
        """Get labels for each dimension in a reduction method.

        Parameters
        ----------
        dr_key
            The dimension reduction key

        Returns
        -------
        List of labels for each dimension

        """
        dr_label = self.get_reduction_label(dr_key)
        num_dims = self.adata.obsm[dr_key].shape[1]
        return [f"{dr_label}{i + 1}" for i in range(num_dims)]

    def create_plot(
        self,
        *,
        dr_key: str,
        x_value: str,
        y_value: str,
        color_by_dim: Literal["obs", "cols"],
        color_by: str,
        datashade_value: bool,
        show_labels: bool,
        cmap: list[str] | str,
    ) -> pn.viewable.Viewable:
        """Create a manifold map plot with the specified parameters.

        Parameters
        ----------
        dr_key
            Dimensionality reduction key
        x_value
            X-axis dimension label
        y_value
            Y-axis dimension label
        color_by_dim
            Dimension to use for coloring
        color_by
            Variable to use for coloring
        datashade_value
            Whether to enable datashading
        show_labels
            Whether to show labels
        cmap
            Colormap

        Returns
        -------
        The plot or an error message

        """
        x_data = self.adata.obsm[dr_key]
        dr_label = self.get_reduction_label(dr_key)

        if x_value == y_value:
            return pmui.pane.Typography(
                "Please select different dimensions for X and Y axes."
            )

        # Extract indices from dimension labels
        try:
            x_dim = int(x_value.replace(dr_label, "")) - 1
            y_dim = int(y_value.replace(dr_label, "")) - 1
        except (ValueError, AttributeError):
            return pmui.pane.Typography(
                f"Error parsing dimensions. "
                f"Make sure to select valid {dr_label} dimensions."
            )

        if color_by_dim == "obs":
            color_data = self.adata.obs[color_by].values
        elif color_by_dim == "cols":
            color_data = self.adata.obs_vector(self._get_var())
        else:
            msg = "color_by_dim must be obs or cols"
            raise ValueError(msg)

        # Configure the plot
        config = ManifoldMapConfig(
            width=self.width,
            height=self.height,
            datashading=datashade_value,
            show_labels=show_labels,
            title=f"{dr_label}.{color_by}",
            cmap=cmap,
            responsive=self.responsive,
        )

        self.plot = create_manifoldmap_plot(
            x_data,
            color_data,
            x_dim,
            y_dim,
            color_by,
            x_value,
            y_value,
            categorical=self._categorical,
            **config,
        )

        return self.plot

    @param.depends(
        # Only include derived parameters to avoid calling create_plot
        # unnecessarily.
        "x_axis",
        "y_axis",
        "colormap",
        "datashade",
        "show_labels",
        "_replot",
    )
    def _plot_view(self) -> pn.viewable.Viewable:
        return self.create_plot(
            dr_key=self.reduction,
            x_value=self.x_axis,
            y_value=self.y_axis,
            color_by_dim=self.color_by_dim,
            color_by=self.color_by,
            datashade_value=self.datashade,
            show_labels=self.show_labels,
            cmap=self.colormap,
        )

    def __panel__(self) -> pn.viewable.Viewable:
        """Create the Panel application layout.

        Returns
        -------
        The assembled panel application

        """
        # Widgets
        color_by_dim = pmui.widgets.RadioButtonGroup.from_param(
            self.param.color_by_dim,
            sizing_mode="stretch_width",
        )
        color = pmui.widgets.AutocompleteInput.from_param(
            self.param.color_by,
            name="",
            min_characters=0,
            search_strategy="includes",
            case_sensitive=False,
            description="",
            sizing_mode="stretch_width",
        )
        stylesheet = """
        label {
            color: rgba(0, 0, 0, 0.6);
        }
        .bk-input {
            border-color: #ccc;
            height: 48px;
        }
        .bk-input:hover {
            border: 1px solid rgba(0, 0, 0, 0.87) !important;
        }
        """
        colormap = pn.widgets.ColorMap.from_param(
            self.param.colormap,
            stylesheets=[stylesheet],
            sizing_mode="stretch_width",
        )
        # Create widget box
        widgets = pmui.Column(
            pmui.widgets.Select.from_param(
                self.param.reduction,
                description="",
                sizing_mode="stretch_width",
            ),
            pmui.widgets.Select.from_param(
                self.param.x_axis,
                sizing_mode="stretch_width",
            ),
            pmui.widgets.Select.from_param(
                self.param.y_axis,
                sizing_mode="stretch_width",
            ),
            color_by_dim,
            color,
            colormap,
            pmui.widgets.Checkbox.from_param(
                self.param.datashade,
                description="",
                sizing_mode="stretch_width",
            ),
            pmui.widgets.Checkbox.from_param(
                self.param.show_labels,
                description="",
                sizing_mode="stretch_width",
            ),
            visible=self.param.show_widgets,
            sx={"border": 1, "borderColor": "#e3e3e3", "borderRadius": 1},
            sizing_mode="stretch_width",
            max_width=400,
        )

        # Return the assembled layout
        return pmui.Row(widgets, self._plot_view)
