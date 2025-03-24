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


def create_featuremap_plot(
    x_data: np.ndarray,
    color_data: np.ndarray,
    x_dim: int,
    y_dim: int,
    color_var: str,
    xaxis_label: str,
    yaxis_label: str,
    width: int = 300,
    height: int = 300,
    datashading: bool = True,
    labels: bool = False,
    cont_cmap: str = "viridis",
    cat_cmap: list = cc.b_glasbey_category10,
    title: str = "",
) -> hv.Element:
    """Create a comprehensive feature map plot with options for datashading and labels

    Parameters
    ----------
    - x_data: numpy.ndarray, shape n_obs by n_dimensions
    - color_data: numpy.ndarray, shape n_obs color values (categorical or continuous)
    - x_dim, y_dim: int, indices to use as x or y data
    - color_var: str, name to give the coloring dimension
    - xaxis_label, yaxis_label: str, labels for the axes
    - width, height: int, dimensions of the plot
    - datashading: bool, whether to apply datashader
    - labels: bool, whether to overlay labels at median positions
    - cont_cmap: str or list, colormap for continuous data
    - cat_cmap: str or list, colormap for categorical data

    """
    is_categorical = (
        color_data.dtype.name in ["category", "categorical", "bool"]
        or np.issubdtype(color_data.dtype, np.object_)
        or np.issubdtype(color_data.dtype, np.str_)
    )

    # Set colormap and plot options based on data type
    if is_categorical:
        n_unq_cat = len(np.unique(color_data))
        # hack so that cat cmap doesn't stretch and skip colors
        cmap = cat_cmap[:n_unq_cat]
        colorbar = False
        if labels:
            show_legend = False
        else:
            show_legend = True
    else:
        cmap = cont_cmap
        show_legend = False
        colorbar = True

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

    # Apply datashading if requested
    if datashading:
        if is_categorical:
            # For categorical data, count by category
            aggregator = ds.count_cat(color_var)
            plot = hd.rasterize(plot, aggregator=aggregator)
            plot = hd.dynspread(plot, threshold=0.5)
            plot = plot.opts(cmap=cmap, tools=["hover"])

            if labels:
                # Add labels at median positions
                unique_categories = np.unique(color_data)
                labels_data = []
                for cat in unique_categories:
                    mask = color_data == cat
                    median_x = np.median(x_data[mask, x_dim])
                    median_y = np.median(x_data[mask, y_dim])
                    labels_data.append((median_x, median_y, str(cat)))
                labels_element = hv.Labels(
                    labels_data, [xaxis_label, yaxis_label], "Label"
                ).opts(**label_opts)
                plot = plot * labels_element
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
        else:
            # For continuous data, take the mean
            aggregator = ds.mean(color_var)
            plot = hd.rasterize(plot, aggregator=aggregator)
            plot = hd.dynspread(plot, threshold=0.5)
            plot = plot.opts(cmap=cmap, colorbar=colorbar)
    else:
        # Standard plot without datashading
        plot = plot.opts(**plot_opts)
        if is_categorical and labels:
            # Add labels for non-datashaded categorical plot
            unique_categories = np.unique(color_data)
            labels_data = []
            for cat in unique_categories:
                mask = color_data == cat
                median_x = np.median(x_data[mask, x_dim])
                median_y = np.median(x_data[mask, y_dim])
                labels_data.append((median_x, median_y, str(cat)))
            labels_element = hv.Labels(
                labels_data, [xaxis_label, yaxis_label], "Label"
            ).opts(**label_opts)
            plot = plot * labels_element

    return plot.opts(
        title=title,
        tools=["hover"],
        show_legend=show_legend,
        frame_width=width,
        frame_height=height,
    )


class FeatureMapApp(pn.viewable.Viewer):
    """Create an interactive feature map application for exploring AnnData objects.

    This application provides widgets to select dimensionality reduction methods,
    dimensions for x and y axes, coloring variables, and display options.
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

    def __init__(self, **params):
        super().__init__(**params)
        self.dr_options = list(self.adata.obsm.keys())
        if not self.reduction:
            self.reduction = self.dr_options[0]
        # self.default_dr = reduction or dr_options[0]

        self.color_options = list(self.adata.obs.columns)
        if not self.color_by:
            self.color_by = (
                "cell_type"
                if "cell_type" in self.color_options
                else self.color_options[0]
            )
        # self.default_color = color_by or color_options[0]

    def get_reduction_label(self, dr_key):
        return dr_key.split("_")[1].upper() if "_" in dr_key else dr_key.upper()

    def get_dim_labels(self, dr_key):
        dr_label = self.get_reduction_label(dr_key)
        num_dims = self.adata.obsm[dr_key].shape[1]
        return [f"{dr_label}{i + 1}" for i in range(num_dims)]

    def create_plot(
        self, dr_key, x_value, y_value, color_value, datashade_value, label_value
    ):
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

        return create_featuremap_plot(
            x_data,
            color_data,
            x_dim,
            y_dim,
            color_value,
            x_value,
            y_value,
            width=self.width,
            height=self.height,
            datashading=datashade_value,
            labels=label_value,
            title=f"{dr_label}.{color_value}",
        )

    def __panel__(self):
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

        # reset dim options when reduction selection changes
        @hold()
        def reset_dimension_options(event):
            new_dims = self.get_dim_labels(event.new)
            x_axis.param.update(options=new_dims, value=new_dims[0])
            y_axis.param.update(options=new_dims, value=new_dims[1])

        dr_select.param.watch(reset_dimension_options, "value")

        plot_pane = pn.bind(
            self.create_plot,
            dr_key=dr_select,
            x_value=x_axis,
            y_value=y_axis,
            color_value=color,
            datashade_value=datashade_switch,
            label_value=label_switch,
        )

        widgets = pn.WidgetBox(
            dr_select,
            x_axis,
            y_axis,
            color,
            datashade_switch,
            label_switch,
            visible=self.show_widgets,
        )

        return pn.Row(widgets, plot_pane)
