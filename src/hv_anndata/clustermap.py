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
    expression_data: np.ndarray,
    obs_names: pd.Index,
    var_names: pd.Index,
    color_data: np.ndarray | None = None,
    color_by: str | None = None,
    *,
    categorical: bool | None = None,
    **config: Unpack[ClusterMapConfig],
) -> hv.Element:
    """Create a hierarchically-clustered heatmap using HoloViews.

    Parameters
    ----------
    expression_data
        Matrix with shape n_obs by n_vars containing expression values
    obs_names
        Names for observations (cells)
    var_names
        Names for variables (genes)
    color_data
        Optional array with shape n_obs containing color values for annotations
    color_by
        Name to give the coloring dimension
    categorical
        Whether the color_data is categorical
    config
        Additional configuration options, see :class:`ClusterMapConfig`

    Returns
    -------
    HoloViews element with the clustered heatmap

    """
    # Extract config with defaults
    width = config.get("width", 600)
    height = config.get("height", 400)
    cmap = config.get("cmap", "viridis")
    title = config.get("title", "")
    colorbar = config.get("colorbar", True)

    # Create DataFrame
    df = pd.DataFrame(expression_data, index=obs_names, columns=var_names)

    # Convert to long format for HoloViews HeatMap
    df_melted = df.reset_index().melt(
        id_vars="index", var_name="gene", value_name="expression"
    )
    df_melted.rename(columns={"index": "cell"}, inplace=True)

    # Add categorical annotation if provided
    vdims = ["expression"]
    if color_data is not None and color_by is not None:
        # Determine if color data is categorical
        if categorical is None:
            categorical = _is_categorical(color_data)

        # Create mapping from cell names to color values
        color_mapping = dict(zip(obs_names, color_data, strict=False))
        df_melted[color_by] = df_melted["cell"].map(color_mapping)
        vdims.append(color_by)

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

    clustered_plot = clustered_plot.opts(
        hv.opts.HeatMap(**plot_opts), hv.opts.Dendrogram(xaxis=None, yaxis=None)
    )

    return clustered_plot


def clustermap_hv(adata, obs_keys=None, use_raw=None, max_genes=None, **kwds):
    """Create a hierarchically-clustered heatmap using HoloViews.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    obs_keys : str, optional
        Categorical annotation to plot with different colors.
        Currently, only a single key is supported.
    use_raw : bool, optional
        Whether to use `raw` attribute of `adata`.
        Defaults to `True` if `.raw` is present.
    max_genes : int, optional
        Maximum number of genes to include in the heatmap.
        If None, all genes are included.
    **kwds
        Additional keyword arguments passed to the plot configuration.

    Returns
    -------
    HoloViews Layout object containing the clustered heatmap with dendrograms.

    """
    # Determine whether to use raw data
    if use_raw is None:
        use_raw = adata.raw is not None

    # Extract data matrix
    X = adata.raw.X if use_raw else adata.X

    # Convert sparse matrix to dense if needed
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Filter genes if max_genes is specified
    var_names = adata.var_names
    if max_genes is not None and len(var_names) > max_genes:
        # Select genes with highest variance for better clustering
        gene_vars = np.var(X, axis=0)
        top_gene_indices = np.argsort(gene_vars)[-max_genes:]
        X = X[:, top_gene_indices]
        var_names = var_names[top_gene_indices]

    # Prepare color data if obs_keys is provided
    color_data = None
    if obs_keys is not None:
        if obs_keys not in adata.obs.columns:
            raise ValueError(f"obs_keys '{obs_keys}' not found in adata.obs")
        color_data = adata.obs[obs_keys].values

    # Create the plot using the parameterized function
    return create_clustermap_plot(
        expression_data=X,
        obs_names=adata.obs_names,
        var_names=var_names,
        color_data=color_data,
        color_by=obs_keys,
        **kwds,
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

        # Set up use_raw default
        if self.use_raw is None:
            self.use_raw = self.adata.raw is not None

    @param.depends("obs_keys", watch=True, on_init=True)
    def _update_on_obs_keys(self) -> None:
        # TODO investigate:
        if not self.obs_keys:
            return
        old_is_categorical = self._categorical
        color_data = self.adata.obs[self.obs_keys].values
        self._categorical = _is_categorical(color_data)
        if old_is_categorical != self._categorical or not self.cmap:
            cmaps = CAT_CMAPS if self._categorical else CONT_CMAPS
            self.param.cmap.objects = cmaps
            self.cmap = list(cmaps.values())[0]
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

        return clustermap_hv(
            self.adata,
            obs_keys=obs_keys,
            use_raw=use_raw,
            max_genes=max_genes,
            **config,
        )

    def __panel__(self) -> pn.viewable.Viewable:
        """Create the Panel application layout.

        Returns
        -------
        The assembled panel application

        """
        hv_pane = pn.pane.HoloViews()
        hv_pane.object = pn.bind(
            self.create_plot,
            obs_keys=self.param.obs_keys,
            use_raw=self.param.use_raw,
            cmap=self.param.cmap,
            max_genes=self.param.max_genes,
        )
        # Return the assembled layout
        return pmui.Row(self._widgets, hv_pane)
