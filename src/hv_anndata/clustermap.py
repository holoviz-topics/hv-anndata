"""Interactive hierarchically-clustered heatmap visualization for AnnData objects."""

from __future__ import annotations

from typing import TypedDict, Unpack

import anndata as ad
import bokeh.palettes
import colorcet as cc
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import panel_material_ui as pmui
import param
from holoviews.operation import dendrogram
from panel.reactive import hold

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


def _is_categorical(arr: np.ndarray) -> bool:
    return (
        arr.dtype.name in ["category", "categorical", "bool"]
        or np.issubdtype(arr.dtype, np.object_)
        or np.issubdtype(arr.dtype, np.str_)
    )


class ClusterMapConfig(TypedDict, total=False):
    """Configuration options for cluster map plotting."""

    width: int
    """width of the plot (default: 600)"""
    height: int
    """height of the plot (default: 400)"""
    cmap: str | list[str]
    """cmap for the heatmap"""
    title: str
    """plot title (default: "")"""
    colorbar: bool
    """whether to show colorbar (default: True)"""
    show_legend: bool
    """whether to show legend for categorical data (default: True)"""


def create_clustermap_plot(
    adata: ad.AnnData,
    obs_keys: str | None = None,
    use_raw: bool | None = None,  # noqa: FBT001
    max_genes: int | None = None,
    **config: Unpack[ClusterMapConfig],
) -> hv.Layout:
    """Create a hierarchically-clustered heatmap using HoloViews.

    Parameters
    ----------
    adata
        Annotated data matrix
    obs_keys
        Categorical annotation to plot with different colors.
        Currently, only a single key is supported.
    use_raw
        Whether to use `raw` attribute of `adata`.
        Defaults to `True` if `.raw` is present.
    max_genes
        Maximum number of genes to include in the heatmap.
        If None, all genes are included.
    config
        Additional configuration options, see :class:`ClusterMapConfig`

    Returns
    -------
    HoloViews Layout object containing the clustered heatmap with dendrograms

    """
    # Determine whether to use raw data
    if use_raw is None:
        use_raw = adata.raw is not None

    # Extract data matrix
    x = adata.raw.X if use_raw else adata.X

    # Convert sparse matrix to dense if needed
    if hasattr(x, "toarray"):
        x = x.toarray()

    # Filter genes if max_genes is specified
    var_names = adata.var_names
    if max_genes is not None and len(var_names) > max_genes:
        # Select genes with highest variance for better clustering
        gene_vars = np.var(x, axis=0)
        top_gene_indices = np.argsort(gene_vars)[-max_genes:]
        x = x[:, top_gene_indices]
        var_names = var_names[top_gene_indices]

    # Prepare color data if obs_keys is provided
    color_data = None
    if obs_keys is not None:
        if obs_keys not in adata.obs.columns:
            err_msg = f"obs_keys '{obs_keys}' not found in adata.obs"
            raise ValueError(err_msg)
        color_data = adata.obs[obs_keys].values

    # Extract config with defaults
    width = config.get("width", 600)
    height = config.get("height", 400)
    cmap = config.get("cmap", "viridis")
    title = config.get("title", "")
    colorbar = config.get("colorbar", True)

    # Create DataFrame
    df = pd.DataFrame(x, index=adata.obs_names, columns=var_names)

    # Convert to long format for HoloViews HeatMap
    df_melted = df.reset_index().melt(
        id_vars="index", var_name="gene", value_name="expression"
    )
    df_melted = df_melted.rename(columns={"index": "cell"})

    # Add categorical annotation if provided
    vdims = ["expression"]
    if color_data is not None and obs_keys is not None:
        # Create mapping from cell names to color values
        color_mapping = dict(zip(adata.obs_names, color_data, strict=False))
        df_melted[obs_keys] = df_melted["cell"].map(color_mapping)
        vdims.append(obs_keys)

    # Create base heatmap
    heatmap = hv.HeatMap(df_melted, kdims=["gene", "cell"], vdims=vdims)

    # Apply clustering with dendrograms
    clustered_plot = dendrogram(
        heatmap, main_dim="expression", adjoint_dims=["gene", "cell"]
    )

    # Configure plot options
    plot_opts = {
        "colorbar": colorbar,
        "tools": ["hover"],
        "width": width,
        "height": height,
        "cmap": cmap,
        "xrotation": 90,
        "yaxis": None,
        "show_grid": False,
        "title": title,
    }

    return clustered_plot.opts(
        hv.opts.HeatMap(**plot_opts), hv.opts.Dendrogram(xaxis=None, yaxis=None)
    )


class ClusterMap(pn.viewable.Viewer):
    """Interactive cluster map application for exploring AnnData objects.

    This application provides widgets to select coloring variables and display options
    for hierarchically-clustered heatmaps.

    Parameters
    ----------
    adata
        AnnData object to visualize
    use_raw
        Whether to use raw data from adata
    obs_keys
        Initial observation key to use for coloring
    color_by_dim
        Color by dimension, one of 'obs' (default) or 'cols'
    cmap
        Initial cmap to use
    width
        Width of the plot
    height
        Height of the plot
    show_widgets
        Whether to show control widgets

    """

    adata: ad.AnnData = param.ClassSelector(
        class_=ad.AnnData, doc="AnnData object to visualize"
    )
    use_raw: bool = param.Boolean(
        default=None, allow_None=True, doc="Whether to use raw data from adata"
    )
    obs_keys: str = param.Selector(doc="Observation key for coloring")
    color_by_dim: str = param.Selector(
        default="obs",
        objects={"Observations": "obs", "Variables": "cols"},
        label="Color By",
    )
    cmap: str = param.Selector()
    width: int = param.Integer(default=600, doc="Width of the plot")
    height: int = param.Integer(default=400, doc="Height of the plot")
    max_genes: int = param.Integer(
        default=50,
        allow_None=True,
        bounds=(20, 100),
        doc="Maximum number of genes to include in the heatmap",
    )
    show_widgets: bool = param.Boolean(
        default=True, doc="Whether to show control widgets"
    )
    _replot: bool = param.Event()

    def __init__(self, **params: object) -> None:
        """Initialize the ClusterMap with the given parameters."""
        # Widgets
        super().__init__(**params)
        self._widgets = pmui.Column(
            pmui.widgets.Select.from_param(
                self.param.obs_keys,
                description="",
                sizing_mode="stretch_width",
            ),
            pn.widgets.ColorMap.from_param(
                self.param.cmap,
                sizing_mode="stretch_width",
            ),
            pmui.widgets.Checkbox.from_param(
                self.param.use_raw,
                description="",
                sizing_mode="stretch_width",
            ),
            visible=self.param.show_widgets,
            sx={"border": 1, "borderColor": "#e3e3e3", "borderRadius": 1},
            sizing_mode="stretch_width",
            max_width=400,
        )

        self._categorical = False

        # Set up observation key options
        obs_options = list(self.adata.obs.columns)
        self.param["obs_keys"].objects = obs_options
        if not self.obs_keys:
            if DEFAULT_COLOR_BY in obs_options:
                self.obs_keys = DEFAULT_COLOR_BY
            else:
                self.obs_keys = obs_options[0]
        # Initialize colormap based on first obs_keys selection
        if self.obs_keys:
            color_data = self.adata.obs[self.obs_keys].values
            self._categorical = _is_categorical(color_data)
            cmaps = CAT_CMAPS if self._categorical else CONT_CMAPS
            self.param.cmap.objects = cmaps
            if not self.cmap:
                self.cmap = next(iter(cmaps.values()))

        # Set up use_raw default
        if self.use_raw is None:
            self.use_raw = self.adata.raw is not None

    @hold()
    @param.depends("obs_keys", watch=True)
    def _update_on_obs_keys(self) -> None:
        if not self.obs_keys:
            return
        old_is_categorical = self._categorical
        color_data = self.adata.obs[self.obs_keys].values
        self._categorical = _is_categorical(color_data)
        if old_is_categorical != self._categorical or not self.cmap:
            cmaps = CAT_CMAPS if self._categorical else CONT_CMAPS
            self.param.cmap.objects = cmaps
            self.cmap = next(iter(cmaps.values()))
        self._replot = True

    def create_plot(
        self,
        *,
        obs_keys: str,
        use_raw: bool,
        cmap: list[str] | str,
        max_genes: int | None,
    ) -> pn.viewable.Viewable:
        """Create a cluster map plot with the specified parameters.

        Parameters
        ----------
        obs_keys
            Observation key for coloring
        use_raw
            Whether to use raw data
        cmap
            cmap
        max_genes
            Maximum number of genes to include

        Returns
        -------
        The clustered heatmap plot

        """
        config = ClusterMapConfig(
            width=self.width,
            height=self.height,
            cmap=cmap,
            title=f"Clustered Heatmap - {obs_keys}",
        )

        return create_clustermap_plot(
            self.adata,
            obs_keys=obs_keys,
            use_raw=use_raw,
            max_genes=max_genes,
            **config,
        )

    @param.depends(
        # Wrapper to fix could not be replaced with new model GridPlot(...),
        # ensure that the parent is not modified at the same time the panel
        # is being updated.
        "obs_keys",
        "use_raw",
        "cmap",
        "max_genes",
        "_replot",
    )
    def _plot_view(self) -> pn.viewable.Viewable:
        """Create the plot view with parameter dependencies."""
        return self.create_plot(
            obs_keys=self.obs_keys,
            use_raw=self.use_raw,
            cmap=self.cmap,
            max_genes=self.max_genes,
        )

    def __panel__(self) -> pn.viewable.Viewable:
        """Create the Panel application layout.

        Returns
        -------
        The assembled panel application

        """
        return pmui.Row(self._widgets, self._plot_view)
