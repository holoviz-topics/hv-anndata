"""Interactive vizualization of AnnData dimension reductions with HoloViews and Panel."""  # noqa: E501

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack

import anndata as ad
import colorcet as cc
import datashader as ds
import holoviews as hv
import holoviews.operation.datashader as hd
import numpy as np
import panel as pn
import param
from panel.reactive import hold

if TYPE_CHECKING:
    from collections.abc import Sequence


DEFAULT_COLOR_BY = "cell_type"


class ManifoldMapConfig(TypedDict, total=False):
    """Configuration options for manifold map plotting."""

    width: int
    """width of the plot (default: 300)"""
    height: int
    """height of the plot (default: 300)"""
    datashading: bool
    """whether to apply datashader (default: True)"""
    labels: bool
    """whether to overlay labels at median positions (default: False)"""
    cont_cmap: str
    """colormap for continuous data (default: "viridis")"""
    cat_cmap: list
    """colormap for categorical data (default: cc.b_glasbey_category10)"""
    title: str
    """plot title (default: "")"""


def create_manifoldmap_plot(
    x_data: np.ndarray,
    color_data: np.ndarray,
    x_dim: int,
    y_dim: int,
    color_var: str,
    xaxis_label: str,
    yaxis_label: str,
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
    color_var
        Name to give the coloring dimension
    xaxis_label
        Label for the x axis
    yaxis_label
        Label for the y axis
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
    labels = config.get("labels", False)
    cont_cmap = config.get("cont_cmap", "viridis")
    cat_cmap = config.get("cat_cmap", cc.b_glasbey_category10)
    title = config.get("title", "")

    # Determine if color data is categorical
    is_categorical = (
        color_data.dtype.name in ["category", "categorical", "bool"]
        or np.issubdtype(color_data.dtype, np.object_)
        or np.issubdtype(color_data.dtype, np.str_)
    )

    # Set colormap and plot options based on data type
    if is_categorical:
        n_unq_cat = len(np.unique(color_data))
        # Use subset of categorical colormap to preserve distinct colors
        cmap = cat_cmap[:n_unq_cat]
        colorbar = False
        show_legend = not labels
    else:
        cmap = cont_cmap
        show_legend = False
        colorbar = True

    # Create basic plot
    plot = hv.Points(
        (x_data[:, x_dim], x_data[:, y_dim], color_data),
        [xaxis_label, yaxis_label],
        color_var,
    )

    # Options for standard (non-datashaded) plot
    plot_opts = dict(
        color=color_var,
        cmap=cmap,
        size=1,
        alpha=0.5,
        colorbar=colorbar,
        padding=0,
        tools=["hover", "box_select", "lasso_select"],
        show_legend=show_legend,
        legend_position="right",
    )

    # Options for labels
    label_opts = dict(text_font_size="8pt", text_color="black")

    # Apply different rendering based on configuration
    if not datashading:
        # Standard plot without datashading
        plot = plot.opts(**plot_opts)

        # Add labels if categorical and requested
        if is_categorical and labels:
            plot = _add_category_labels(
                plot,
                x_data,
                color_data,
                x_dim,
                y_dim,
                xaxis_label,
                yaxis_label,
                label_opts,
            )
    # Apply datashading with different approaches for categorical vs continuous
    elif is_categorical:
        plot = _apply_categorical_datashading(
            plot,
            x_data=x_data,
            color_data=color_data,
            x_dim=x_dim,
            y_dim=y_dim,
            color_var=color_var,
            cmap=cmap,
            xaxis_label=xaxis_label,
            yaxis_label=yaxis_label,
            labels=labels,
            label_opts=label_opts,
        )
    else:
        # For continuous data, take the mean
        aggregator = ds.mean(color_var)
        plot = hd.rasterize(plot, aggregator=aggregator)
        plot = hd.dynspread(plot, threshold=0.5)
        plot = plot.opts(cmap=cmap, colorbar=colorbar)

    # Apply final options to the plot
    return plot.opts(
        title=title,
        tools=["hover", "box_select", "lasso_select"],
        show_legend=show_legend,
        frame_width=width,
        frame_height=height,
    )


def _add_category_labels(  # noqa: PLR0913
    plot: hv.Element,
    x_data: np.ndarray,
    color_data: np.ndarray,
    x_dim: int,
    y_dim: int,
    xaxis_label: str,
    yaxis_label: str,
    label_opts: dict[str, Any],
) -> hv.Element:
    """Add category labels to a plot.

    Parameters
    ----------
    plot
        The base plot to add labels to
    x_data
        Coordinate data
    color_data
        Category data for coloring
    x_dim
        Index for x dimension
    y_dim
        Index for y dimension
    xaxis_label
        X-axis label
    yaxis_label
        Y-axis label
    label_opts
        Options for label formatting

    Returns
    -------
    Plot with labels added

    """
    unique_categories = np.unique(color_data)
    labels_data = []

    for cat in unique_categories:
        mask = color_data == cat
        if np.any(mask):
            median_x = np.median(x_data[mask, x_dim])
            median_y = np.median(x_data[mask, y_dim])
            labels_data.append((median_x, median_y, str(cat)))

    labels_element = hv.Labels(labels_data, [xaxis_label, yaxis_label], "Label").opts(
        **label_opts
    )

    return plot * labels_element


def _apply_categorical_datashading(  # noqa: PLR0913
    plot: hv.Element,
    *,
    x_data: np.ndarray,
    color_data: np.ndarray,
    x_dim: int,
    y_dim: int,
    color_var: str,
    cmap: Sequence[str],
    xaxis_label: str,
    yaxis_label: str,
    labels: bool,
    label_opts: dict[str, Any],
) -> hv.Element:
    """Apply datashading to categorical data.

    Parameters
    ----------
    plot
        The base plot to apply datashading to
    x_data
        Coordinate data
    color_data
        Category data for coloring
    x_dim
        Index for x dimension
    y_dim
        Index for y dimension
    color_var
        Name of the color variable
    cmap
        Colormap to use
    xaxis_label
        X-axis label
    yaxis_label
        Y-axis label
    labels
        Whether to add category labels
    label_opts
        Options for label formatting

    Returns
    -------
    Datashaded plot with optional labels and legend

    """
    # For categorical data, count by category
    aggregator = ds.count_cat(color_var)
    plot = hd.rasterize(plot, aggregator=aggregator)
    plot = hd.dynspread(plot, threshold=0.5)
    plot = plot.opts(cmap=cmap, tools=["hover", "box_select", "lasso_select"])

    # Add either labels or a custom legend
    if labels:
        plot = _add_category_labels(
            plot, x_data, color_data, x_dim, y_dim, xaxis_label, yaxis_label, label_opts
        )
    else:
        # Create a custom legend for datashaded categorical plot
        unique_categories = np.unique(color_data)
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
        plot = plot * legend

    return plot


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
    datashade
        Whether to enable datashading
    width
        Width of the plot
    height
        Height of the plot
    labels
        Whether to show labels
    show_widgets
        Whether to show control widgets

    """

    adata: ad.AnnData = param.ClassSelector(  # type: ignore[assignment]
        class_=ad.AnnData, doc="AnnData object to visualize"
    )
    reduction: str | None = param.String(  # type: ignore[assignment]
        default=None, doc="Dimension reduction method", allow_None=True
    )
    color_by_dim: str = param.Selector(  # type: ignore[assignment]
        default="obs",
        objects={"Observations": "obs", "Genes": "cols"},
    )
    color_by: str = param.Selector(  # type: ignore[assignment]
        doc="Coloring variable"
    )
    datashade: bool = param.Boolean(default=True, doc="Whether to enable datashading")  # type: ignore[assignment]
    width: int = param.Integer(default=300, doc="Width of the plot")  # type: ignore[assignment]
    height: int = param.Integer(default=300, doc="Height of the plot")  # type: ignore[assignment]
    labels: bool = param.Boolean(default=False, doc="Whether to show labels")  # type: ignore[assignment]
    show_widgets: bool = param.Boolean(  # type: ignore[assignment]
        default=True, doc="Whether to show control widgets"
    )
    _color_info: tuple = param.Tuple(length=2)  # type: ignore[assignment]

    def __init__(self, **params: object) -> None:
        """Initialize the ManifoldMapApp with the given parameters."""
        super().__init__(**params)
        self.dr_options = list(self.adata.obsm.keys())
        if not self.reduction:
            self.reduction = self.dr_options[0]

        self._color_options = {
            "obs": list(self.adata.obs.columns),
            "cols": list(self.adata.var_names),
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
            self._update_color_info()

    @param.depends("color_by_dim", watch=True)
    def _on_color_by_dim(self) -> None:
        values = self._color_options[self.color_by_dim]
        self.param.color_by.objects = values
        self.color_by = values[0]

    @param.depends("color_by", watch=True)
    def _update_color_info(self) -> None:
        self._color_info = (self.color_by_dim, self.color_by)

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
        color_info: tuple[Literal["obs", "cols"], str],
        datashade_value: bool,
        label_value: bool,
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
        color_info
            Dimension and variable to use for coloring
        datashade_value
            Whether to enable datashading
        label_value
            Whether to show labels

        Returns
        -------
        The plot or an error message

        """
        x_data = self.adata.obsm[dr_key]
        dr_label = self.get_reduction_label(dr_key)

        if x_value == y_value:
            return pn.pane.Markdown(
                "Please select different dimensions for X and Y axes."
            )

        # Extract indices from dimension labels
        try:
            x_dim = int(x_value.replace(dr_label, "")) - 1
            y_dim = int(y_value.replace(dr_label, "")) - 1
        except (ValueError, AttributeError):
            return pn.pane.Markdown(
                f"Error parsing dimensions. "
                f"Make sure to select valid {dr_label} dimensions."
            )

        color_dim, color_key = color_info
        if color_dim == "obs":
            color_data = self.adata.obs[color_key].values
        elif color_dim == "cols":
            color_data = self.adata.obs_vector(color_key)
        else:
            msg = "color_dim must be obs or cols"
            raise ValueError(msg)

        # Configure the plot
        config = ManifoldMapConfig(
            width=self.width,
            height=self.height,
            datashading=datashade_value,
            labels=label_value,
            title=f"{dr_label}.{color_key}",
        )

        return create_manifoldmap_plot(
            x_data,
            color_data,
            x_dim,
            y_dim,
            color_key,
            x_value,
            y_value,
            **config,
        )

    def __panel__(self) -> pn.viewable.Viewable:
        """Create the Panel application layout.

        Returns
        -------
        The assembled panel application

        """
        # Widgets
        dr_select = pn.widgets.Select.from_param(
            self.param.reduction, options=self.dr_options
        )
        initial_dims = self.get_dim_labels(dr_select.value)
        x_axis = pn.widgets.Select(
            name="X-axis", options=initial_dims, value=initial_dims[0]
        )
        y_axis = pn.widgets.Select(
            name="Y-axis", options=initial_dims, value=initial_dims[1]
        )
        color_dim = pn.widgets.RadioButtonGroup.from_param(
            self.param.color_by_dim,
        )
        color = pn.widgets.AutocompleteInput.from_param(
            self.param.color_by,
            name="",
            min_characters=0,
            search_strategy="includes",
            case_sensitive=False,
            stylesheets=[
                ":host .bk-menu.bk-below {max-height: 200px; overflow-y: auto}"
            ],
        )
        datashade_switch = pn.widgets.Checkbox.from_param(
            self.param.datashade, name="Datashader Rasterize For Large Datasets"
        )
        label_switch = pn.widgets.Checkbox.from_param(
            self.param.labels, name="Overlay Labels For Categorical Coloring"
        )

        # Reset dimension options when reduction selection changes
        @hold()
        def reset_dimension_options(event: object) -> None:
            new_dims = self.get_dim_labels(event.new)
            x_axis.param.update(options=new_dims, value=new_dims[0])
            y_axis.param.update(options=new_dims, value=new_dims[1])

        dr_select.param.watch(reset_dimension_options, "value")

        # Bind the plot creation to widget values
        plot_pane = pn.bind(
            self.create_plot,
            dr_key=dr_select,
            x_value=x_axis,
            y_value=y_axis,
            color_info=self.param["_color_info"],
            datashade_value=datashade_switch,
            label_value=label_switch,
        )

        # Create widget box
        widgets = pn.WidgetBox(
            dr_select,
            x_axis,
            y_axis,
            pn.pane.HTML("<strong>Color by</strong>"),
            color_dim,
            color,
            datashade_switch,
            label_switch,
            visible=self.show_widgets,
        )

        # Return the assembled layout
        return pn.Row(widgets, plot_pane)
