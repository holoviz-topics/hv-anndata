"""Interactive hierarchically-clustered heatmap visualization for AnnData objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import anndata as ad
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import panel_material_ui as pmui
import param
from holoviews.operation import dendrogram

if TYPE_CHECKING:
    from typing import Unpack

    from anndata import AnnData


class ClusterMapConfig(TypedDict, total=False):
    """Configuration options for cluster map plotting."""

    cmap: str
    """colormap for the heatmap (default: 'viridis')"""


def create_clustermap_plot(
    adata: ad.AnnData,
    *,
    use_raw: bool | None = None,
    max_genes: int | None = None,
    **config: Unpack[ClusterMapConfig],
) -> hv.core.layout.AdjointLayout:
    """Create a hierarchically-clustered heatmap using HoloViews.

    Parameters
    ----------
    adata
        Annotated data matrix
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
    HoloViews AdjointLayout object containing the clustered heatmap with dendrograms

    """
    if use_raw is None:
        use_raw = adata.raw is not None

    x = adata.raw.X if use_raw else adata.X

    if hasattr(x, "toarray"):
        x = x.toarray()

    # Filter genes if max_genes is specified
    var_names = adata.var_names
    if max_genes is not None and len(var_names) > max_genes:
        gene_vars = np.var(x, axis=0)
        top_gene_indices = np.argsort(gene_vars)[-max_genes:]
        x = x[:, top_gene_indices]
        var_names = var_names[top_gene_indices]
        if var_names.name == "index":
            var_names.name = "variable"

    cmap = config.get("cmap", "magma")

    df = pd.DataFrame(x, index=adata.obs_names, columns=var_names)
    index_name = df.index.name or "index"
    var_name = var_names.name or "variable"

    # Convert to long format for HeatMap
    df_melted = df.reset_index().melt(
        id_vars=index_name, var_name=var_name, value_name="expression"
    )

    main_heatmap = hv.HeatMap(
        df_melted, kdims=[var_name, index_name], vdims=["expression"]
    )

    clustered_plot = dendrogram(
        main_heatmap, main_dim="expression", adjoint_dims=[var_name, index_name]
    )

    main_plot_opts = {
        "cmap": cmap,
        "xrotation": 90,
        "yaxis": None,
        "show_grid": False,
        "tools": ["hover"],
        "colorbar": True,
    }

    return clustered_plot.opts(
        hv.opts.HeatMap(**main_plot_opts), hv.opts.Dendrogram(xaxis=None, yaxis=None)
    )


class ClusterMap(pn.viewable.Viewer):
    """Interactive cluster map application for exploring AnnData objects."""

    adata: ad.AnnData = param.ClassSelector(
        class_=ad.AnnData, doc="AnnData object to visualize"
    )
    use_raw: bool = param.Boolean(
        default=None, allow_None=True, doc="Whether to use raw data from adata"
    )
    cmap: str = param.Selector(
        default="viridis",
        objects=[
            "viridis",
            "magma",
            "plasma",
            "fire",
            "blues",
            "RdBu_r",
            "coolwarm",
            "seismic",
            "bwr",
        ],
        doc="Colormap for expression heatmap",
        label="Expression Colormap",
    )
    max_genes: int = param.Integer(
        default=50,
        allow_None=True,
        bounds=(1, 100),
        doc="Maximum number of genes to include in the heatmap",
    )
    show_widgets: bool = param.Boolean(
        default=True, doc="Whether to show control widgets"
    )
    plot_opts: dict = param.Dict(
        default={}, doc="HoloViews plot options for the clustermap plot"
    )

    def __init__(self, adata: AnnData | None = None, **params: object) -> None:
        """Initialize the ClusterMap with the given parameters."""
        if adata is not None:
            params["adata"] = adata
        super().__init__(**params)

        # Check if raw data exists
        has_raw = self.adata.raw is not None
        if self.use_raw is None:
            self.use_raw = has_raw

        use_raw_widget = pmui.widgets.Checkbox.from_param(
            self.param.use_raw,
            description="Use Raw Data" + ("" if has_raw else " (not available)"),
            sizing_mode="stretch_width",
            disabled=not has_raw,  # Disable if no raw data
        )

        self._widgets = pmui.Column(
            pmui.widgets.Select.from_param(
                self.param.cmap,
                description="Expression Colormap",
                sizing_mode="stretch_width",
            ),
            pmui.widgets.IntSlider.from_param(
                self.param.max_genes,
                description="Max Genes",
                sizing_mode="stretch_width",
            ),
            use_raw_widget,
            visible=self.param.show_widgets,
            sx={"border": 1, "borderColor": "#e3e3e3", "borderRadius": 1},
            sizing_mode="stretch_width",
            max_width=400,
        )

    @param.depends("use_raw", "cmap", "max_genes", "plot_opts")
    def _plot_view(self) -> pn.viewable.Viewable:
        """Create the plot view with parameter dependencies."""
        config = ClusterMapConfig(
            cmap=self.cmap,
        )

        plot = create_clustermap_plot(
            self.adata,
            use_raw=self.use_raw,
            max_genes=self.max_genes,
            **config,
        )

        # Apply user-provided plot options to the HeatMap only
        if self.plot_opts:
            plot = plot.opts(hv.opts.HeatMap(**self.plot_opts))

        return plot

    def __panel__(self) -> pn.viewable.Viewable:
        """Create the Panel application layout."""
        return pmui.Row(self._widgets, self._plot_view)
