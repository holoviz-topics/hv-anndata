"""Interactive visualization of AnnData dimension reductions with HoloViews and Panel."""

from __future__ import annotations

import anndata as ad
import colorcet as cc
import datashader as ds
import holoviews as hv
import holoviews.operation.datashader as hd
import numpy as np
import panel as pn
import param
from panel.reactive import hold
from typing import Any, Dict, List, Optional, Tuple, Union, TypedDict


class FeatureMapConfig(TypedDict, total=False):
    """Configuration options for feature map plotting."""

    width: int
    height: int
    datashading: bool
    labels: bool
    cont_cmap: str
    cat_cmap: list
    title: str


def create_featuremap_plot(
    x_data: np.ndarray,
    color_data: np.ndarray,
    x_dim: int,
    y_dim: int,
    color_var: str,
    xaxis_label: str,
    yaxis_label: str,
    **config: Any,
) -> hv.Element:
    """Create a comprehensive feature map plot with options for datashading and labels.

    Parameters
    ----------
    x_data : np.ndarray
        Array with shape n_obs by n_dimensions containing coordinates
    color_data : np.ndarray
        Array with shape n_obs containing color values (categorical or continuous)
    x_dim : int
        Index to use for x-axis data
    y_dim : int
        Index to use for y-axis data
    color_var : str
        Name to give the coloring dimension
    xaxis_label : str
        Label for the x axis
    yaxis_label : str
        Label for the y axis
    **config : Any
        Additional configuration options including:
        - width: int, width of the plot (default: 300)
        - height: int, height of the plot (default: 300)
        - datashading: bool, whether to apply datashader (default: True)
        - labels: bool, whether to overlay labels at median positions (default: False)
        - cont_cmap: str, colormap for continuous data (default: "viridis")
        - cat_cmap: list, colormap for categorical data (default: cc.b_glasbey_category10)
        - title: str, plot title (default: "")

    Returns
    -------
    hv.Element
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
        tools=["hover"],
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
                plot, x_data, color_data, x_dim, y_dim, 
                xaxis_label, yaxis_label, label_opts
            )
    else:
        # Apply datashading with different approaches for categorical vs continuous
        if is_categorical:
            plot = _apply_categorical_datashading(
                plot, x_data, color_data, x_dim, y_dim, color_var, cmap,
                xaxis_label, yaxis_label, labels, label_opts
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
        tools=["hover"],
        show_legend=show_legend,
        frame_width=width,
        frame_height=height,
    )


def _add_category_labels(
    plot: hv.Element,
    x_data: np.ndarray,
    color_data: np.ndarray,
    x_dim: int,
    y_dim: int,
    xaxis_label: str,
    yaxis_label: str,
    label_opts: Dict[str, Any],
) -> hv.Element:
    """Add category labels to a plot.

    Parameters
    ----------
    plot : hv.Element
        The base plot to add labels to
    x_data : np.ndarray
        Coordinate data
    color_data : np.ndarray
        Category data for coloring
    x_dim : int
        Index for x dimension
    y_dim : int
        Index for y dimension
    xaxis_label : str
        X-axis label
    yaxis_label : str
        Y-axis label
    label_opts : Dict[str, Any]
        Options for label formatting

    Returns
    -------
    hv.Element
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
            
    labels_element = hv.Labels(
        labels_data, [xaxis_label, yaxis_label], "Label"
    ).opts(**label_opts)
    
    return plot * labels_element


def _apply_categorical_datashading(
    plot: hv.Element,
    x_data: np.ndarray,
    color_data: np.ndarray,
    x_dim: int,
    y_dim: int,
    color_var: str,
    cmap: Any,
    xaxis_label: str,
    yaxis_label: str,
    labels: bool,
    label_opts: Dict[str, Any],
) -> hv.Element:
    """Apply datashading to categorical data.

    Parameters
    ----------
    plot : hv.Element
        The base plot to apply datashading to
    x_data : np.ndarray
        Coordinate data
    color_data : np.ndarray
        Category data for coloring
    x_dim : int
        Index for x dimension
    y_dim : int
        Index for y dimension
    color_var : str
        Name of the color variable
    cmap : Any
        Colormap to use
    xaxis_label : str
        X-axis label
    yaxis_label : str
        Y-axis label
    labels : bool
        Whether to add category labels
    label_opts : Dict[str, Any]
        Options for label formatting

    Returns
    -------
    hv.Element
        Datashaded plot with optional labels and legend
    """
    # For categorical data, count by category
    aggregator = ds.count_cat(color_var)
    plot = hd.rasterize(plot, aggregator=aggregator)
    plot = hd.dynspread(plot, threshold=0.5)
    plot = plot.opts(cmap=cmap, tools=["hover"])
    
    # Add either labels or a custom legend
    if labels:
        plot = _add_category_labels(
            plot, x_data, color_data, x_dim, y_dim, 
            xaxis_label, yaxis_label, label_opts
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
                for cat, item in zip(
                    unique_categories, legend_items, strict=False
                )
            }
        ).opts(
            show_legend=True,
            legend_position="right",
            legend_limit=100,
            legend_cols=len(unique_categories) // 10 + 1,
        )
        plot = plot * legend
        
    return plot


class FeatureMapApp(pn.viewable.Viewer):
    """Interactive feature map application for exploring AnnData objects.

    This application provides widgets to select dimensionality reduction methods,
    dimensions for x and y axes, coloring variables, and display options.
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object to visualize
    reduction : Optional[str]
        Initial dimension reduction method to use
    color_by : Optional[str]
        Initial variable to use for coloring
    datashade : bool
        Whether to enable datashading
    width : int
        Width of the plot
    height : int
        Height of the plot
    labels : bool
        Whether to show labels
    show_widgets : bool
        Whether to show control widgets
    """

    adata = param.ClassSelector(class_=ad.AnnData, doc="AnnData object to visualize")
    reduction = param.String(
        default=None, doc="Dimension reduction method", allow_None=True
    )
    color_by = param.String(default=None, doc="Coloring variable", allow_None=True)
    datashade = param.Boolean(default=True, doc="Whether to enable datashading")
    width = param.Integer(default=300, doc="Width of the plot")
    height = param.Integer(default=300, doc="Height of the plot")
    labels = param.Boolean(default=False, doc="Whether to show labels")
    show_widgets = param.Boolean(default=True, doc="Whether to show control widgets")

    def __init__(self, **params: Any) -> None:
        """Initialize the FeatureMapApp with the given parameters."""
        super().__init__(**params)
        self.dr_options = list(self.adata.obsm.keys())
        if not self.reduction:
            self.reduction = self.dr_options[0]

        self.color_options = list(self.adata.obs.columns)
        if not self.color_by:
            self.color_by = (
                "cell_type"
                if "cell_type" in self.color_options
                else self.color_options[0]
            )

    def get_reduction_label(self, dr_key: str) -> str:
        """Get a display label for a dimension reduction key.

        Parameters
        ----------
        dr_key : str
            The dimension reduction key

        Returns
        -------
        str
            A formatted label for display
        """
        return dr_key.split("_")[1].upper() if "_" in dr_key else dr_key.upper()

    def get_dim_labels(self, dr_key: str) -> List[str]:
        """Get labels for each dimension in a reduction method.

        Parameters
        ----------
        dr_key : str
            The dimension reduction key

        Returns
        -------
        List[str]
            List of labels for each dimension
        """
        dr_label = self.get_reduction_label(dr_key)
        num_dims = self.adata.obsm[dr_key].shape[1]
        return [f"{dr_label}{i + 1}" for i in range(num_dims)]

    def create_plot(
        self, 
        dr_key: str, 
        x_value: str, 
        y_value: str, 
        color_value: str, 
        datashade_value: bool, 
        label_value: bool
    ) -> pn.viewable.Viewable:
        """Create a feature map plot with the specified parameters.

        Parameters
        ----------
        dr_key : str
            Dimensionality reduction key
        x_value : str
            X-axis dimension label
        y_value : str
            Y-axis dimension label
        color_value : str
            Variable to use for coloring
        datashade_value : bool
            Whether to enable datashading
        label_value : bool
            Whether to show labels

        Returns
        -------
        pn.viewable.Viewable
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
                f"Error parsing dimensions. Make sure to select valid {dr_label} dimensions."
            )

        # Get color data from .obs or X cols
        try:
            color_data = self.adata.obs[color_value].values
        except KeyError:
            try:
                color_data = (
                    self.adata.X.getcol(self.adata.var_names.get_loc(color_value))
                    .toarray()
                    .flatten()
                )
            except (KeyError, ValueError):
                color_data = np.zeros(self.adata.n_obs)
                print(f"Warning: Could not find {color_value} in obs or var")

        # Configure the plot
        config = FeatureMapConfig(
            width=self.width,
            height=self.height,
            datashading=datashade_value,
            labels=label_value,
            title=f"{dr_label}.{color_value}",
        )

        return create_featuremap_plot(
            x_data,
            color_data,
            x_dim,
            y_dim,
            color_value,
            x_value,
            y_value,
            **config,
        )

    def __panel__(self) -> pn.viewable.Viewable:
        """Create the Panel application layout.

        Returns
        -------
        pn.viewable.Viewable
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
        color = pn.widgets.Select.from_param(
            self.param.color_by, options=self.color_options
        )
        datashade_switch = pn.widgets.Checkbox.from_param(
            self.param.datashade, name="Datashader Rasterize For Large Datasets"
        )
        label_switch = pn.widgets.Checkbox.from_param(
            self.param.labels, name="Overlay Labels For Categorical Coloring"
        )

        # Reset dimension options when reduction selection changes
        @hold()
        def reset_dimension_options(event: Any) -> None:
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
            color_value=color,
            datashade_value=datashade_switch,
            label_value=label_switch,
        )

        # Create widget box
        widgets = pn.WidgetBox(
            dr_select,
            x_axis,
            y_axis,
            color,
            datashade_switch,
            label_switch,
            visible=self.show_widgets,
        )

        # Return the assembled layout
        return pn.Row(widgets, plot_pane)