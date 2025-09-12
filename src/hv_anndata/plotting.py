"""HoloViz plotting using AnnData as input."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import chain
from typing import TYPE_CHECKING, TypedDict

import anndata as ad
import holoviews as hv
import pandas as pd
import panel as pn
import param
from holoviews.core.operation import Operation
from holoviews.core.overlay import Overlay
from holoviews.selection import link_selections
from holoviews.streams import Params
from packaging.version import Version

from .components import GeneSelector

_HOLOVIEWS_VERSION = Version(hv.__version__).release

if TYPE_CHECKING:
    from typing import Any, Literal, NotRequired, Unpack

    from holoviews import DynamicMap, Element

    from .manifoldmap import ManifoldMap


class _DotmapPlotParams(TypedDict):
    kdims: NotRequired[list[str | hv.Dimension]]
    vdims: NotRequired[list[str | hv.Dimension]]
    adata: ad.AnnData
    marker_genes: dict[str, list[str]] | list[str]
    groupby: str
    expression_cutoff: NotRequired[float]
    max_dot_size: NotRequired[int]
    standard_scale: NotRequired[Literal["var", "group", None]]
    use_raw: NotRequired[bool | None]
    mean_only_expressed: NotRequired[bool]
    ls: link_selections


class Dotmap(pn.viewable.Viewer):
    """Create a DotmapPlot from anndata."""

    kdims = param.List(
        default=["marker_line", "cluster"],
        bounds=(2, 2),
        doc="""Key dimensions representing cluster and marker line
        (combined marker cluster name and gene).""",
    )

    vdims = param.List(
        default=[
            "gene_id",
            "mean_expression",
            "percentage",
            "marker_cluster_name",
        ],
        doc="Value dimensions representing expression metrics and metadata.",
    )

    adata = param.ClassSelector(class_=ad.AnnData)
    marker_genes = param.ClassSelector(
        default={}, class_=(dict, list), doc="Dictionary or list of marker genes."
    )
    groupby = param.String(default="cell_type", doc="Column to group by.")
    expression_cutoff = param.Number(default=0.0, doc="Cutoff for expression.")
    max_dot_size = param.Integer(default=20, doc="Maximum size of the dots.")

    standard_scale = param.Selector(
        default=None,
        objects=[None, "var", "group"],
        doc="""\
        Whether to standardize the dimension between 0 and 1. 'var' scales each gene,
        'group' scales each cell type.""",
    )

    use_raw = param.Boolean(
        default=None,
        allow_None=True,
        doc="""\
            Whether to use `.raw` attribute of AnnData if present.

            - None (default): Use `.raw` if available, otherwise use `.X`
            - True: Must use `.raw` attribute (raises error if not available)
            - False: Always use `.X`, ignore `.raw` even if present

            In single-cell analysis, `.raw` typically contains the original
            count data before normalization, while `.X` contains processed data
            (e.g., log-transformed, scaled). Using raw counts is sometimes
            preferred for visualization to show actual expression levels.
            """,
    )

    mean_only_expressed = param.Boolean(
        default=False,
        doc="If True, gene expression is averaged only over expressing cells.",
    )

    def _prepare_data(self) -> pd.DataFrame:  # noqa: C901, PLR0912, PLR0915
        # Flatten the marker_genes preserving order
        is_mapping_marker_genes = isinstance(self.marker_genes, Mapping)
        if is_mapping_marker_genes:
            all_marker_genes = list(
                dict.fromkeys(chain.from_iterable(self.marker_genes.values()))
            )
        else:
            all_marker_genes = list(self.marker_genes)

        # Determine to use raw or processed
        use_raw = self.use_raw
        if use_raw is None:
            use_raw = self.adata.raw is not None
        elif use_raw and self.adata.raw is None:
            err = "use_raw=True but .raw attribute is not present in adata"
            raise ValueError(err)

        # Check which genes are actually present in the correct location
        if use_raw and self.adata.raw is not None:
            available_var_names = self.adata.raw.var_names
        else:
            available_var_names = self.adata.var_names

        missing_genes = set(all_marker_genes) - set(available_var_names)
        if missing_genes:
            print(  # noqa: T201
                f"Warning: The following genes are not present in the dataset and will be skipped: {missing_genes}"  # noqa: E501
            )
            available_marker_genes = [
                g for g in all_marker_genes if g not in missing_genes
            ]
            if not available_marker_genes:
                msg = "None of the specified marker genes are present in the dataset."
                raise ValueError(msg)
        else:
            available_marker_genes = all_marker_genes

        # Subset the data with only available genes
        if use_raw and self.adata.raw is not None:
            adata_subset = self.adata.raw[:, available_marker_genes]
            expression_df = pd.DataFrame(
                adata_subset.X.toarray()
                if hasattr(adata_subset.X, "toarray")
                else adata_subset.X,
                index=self.adata.obs_names,
                columns=available_marker_genes,
            )
        else:
            expression_df = self.adata[:, available_marker_genes].to_df()

        # Join with groupby column
        joined_df = expression_df.join(self.adata.obs[self.groupby])

        def compute_expression(df: pd.DataFrame) -> pd.DataFrame:
            # Separate the groupby column from gene columns
            gene_cols = [col for col in df.columns if col != self.groupby]

            results = {}
            for gene in gene_cols:
                gene_data = df[gene]

                # percentage of expressing cells
                percentage = (gene_data > self.expression_cutoff).mean() * 100

                if self.mean_only_expressed:
                    expressing_mask = gene_data > self.expression_cutoff
                    if expressing_mask.any():
                        mean_expr = gene_data[expressing_mask].mean()
                    else:
                        mean_expr = 0.0
                else:
                    mean_expr = gene_data.mean()

                results[gene] = {"percentage": percentage, "mean_expression": mean_expr}

            return pd.DataFrame(results).T

        grouped = joined_df.groupby(self.groupby, observed=True)
        expression_stats = grouped.apply(compute_expression, include_groups=False)

        if is_mapping_marker_genes:
            data = [  # Likely faster way to do this, but harder to read
                expression_stats.xs(gene, level=1)
                .reset_index(names="cluster")
                .assign(
                    marker_cluster_name=marker_cluster_name,
                    gene_id=gene,
                )
                for marker_cluster_name, gene_list in self.marker_genes.items()
                for gene in gene_list
                # Only include genes that weren't filtered out
                if gene in available_marker_genes
            ]
        else:
            data = [  # Likely faster way to do this, but harder to read
                expression_stats.xs(gene, level=1)
                .reset_index(names="cluster")
                .assign(gene_id=gene)
                for gene in self.marker_genes
                # Only include genes that weren't filtered out
                if gene in available_marker_genes
            ]

        if data:
            df = pd.concat(data, ignore_index=True)
        else:
            df = pd.DataFrame({k: [] for k in self.kdims + self.vdims})

        # Apply standard_scale if specified
        if self.standard_scale == "var":
            # Normalize each gene across all cell types
            for gene in df["gene_id"].unique():
                mask = df["gene_id"] == gene
                gene_data = df.loc[mask, "mean_expression"]
                min_val = gene_data.min()
                max_val = gene_data.max()
                if max_val > min_val:
                    df.loc[mask, "mean_expression"] = (gene_data - min_val) / (
                        max_val - min_val
                    )
                else:
                    df.loc[mask, "mean_expression"] = 0.0

        elif self.standard_scale == "group":
            # Normalize each cell type across all genes
            for cluster in df["cluster"].unique():
                mask = df["cluster"] == cluster
                cluster_data = df.loc[mask, "mean_expression"]
                min_val = cluster_data.min()
                max_val = cluster_data.max()
                if max_val > min_val:
                    df.loc[mask, "mean_expression"] = (cluster_data - min_val) / (
                        max_val - min_val
                    )
                else:
                    df.loc[mask, "mean_expression"] = 0.0

        # Create marker_line column
        if df.empty:
            df["marker_line"] = None
        elif is_mapping_marker_genes:
            df["marker_line"] = df["marker_cluster_name"] + ", " + df["gene_id"]
        else:
            df["marker_line"] = df["gene_id"]
            df["marker_cluster_name"] = None

        return df

    def _get_opts(self) -> dict[str, Any]:
        opts = dict(
            cmap="Reds",
            color=hv.dim("mean_expression"),
            colorbar=True,
            show_legend=False,
            xrotation=45,
        )

        radius_dim = hv.dim("percentage").norm()
        match hv.Store.current_backend:
            case "matplotlib":
                backend_opts = {"s": radius_dim * self.max_dot_size}
            case "bokeh":
                hover_tooltips = [*self.kdims, *self.vdims]
                if "marker_cluster_name" in hover_tooltips and (
                    not isinstance(self.marker_genes, Mapping)
                ):
                    hover_tooltips.remove("marker_cluster_name")
                backend_opts = {
                    "colorbar_position": "left",
                    "clabel": "Mean Expression",
                    "colorbar_opts": {
                        # Does not seem to work
                        "title_text_font_style": "italic",
                    },
                    "min_height": 300,
                    "tools": ["hover"],
                    "line_alpha": 0.2,
                    "line_color": "k",
                    "hover_tooltips": hover_tooltips,
                    "responsive": True,
                }
                if _HOLOVIEWS_VERSION >= (1, 21, 0):
                    backend_opts |= {"radius": radius_dim / 2}
                else:
                    backend_opts |= {"size": radius_dim * self.max_dot_size}
            case _:
                backend_opts = {}

        return opts | backend_opts

    @param.depends("marker_genes")
    def plot(self) -> hv.Points:
        """Plot the Dotmap."""
        df = self._prepare_data()
        plot = hv.Points(df, kdims=self.kdims, vdims=self.vdims)
        plot.opts(**self._get_opts())
        return plot

    def __init__(self, **params: Unpack[_DotmapPlotParams]) -> None:
        """Init the Dotmap."""
        if required := {"adata", "marker_genes", "groupby"} - params.keys():
            msg = f"Needs to have the following argument(s): {required}"
            raise TypeError(msg)
        super().__init__(**params)
        self._widget = GeneSelector(value=self.param.marker_genes)
        self._widget.param.watch(self._update_marker_genes, "value", onlychanged=False)

    def _update_marker_genes(self, event: param.parameterized.Event) -> None:
        # Callback to ensure an event is triggered when the marker_genes dict
        # is only reordered
        trigger = (
            isinstance(self.marker_genes, Mapping) and self.marker_genes == event.new
        )
        self.marker_genes = event.new
        if trigger:
            self.param.trigger("marker_genes")

    def __panel__(self) -> pn.viewable.Viewable:
        return pn.Row(self._widget, self.plot)

    @classmethod
    def from_manifold_map(cls, mm: ManifoldMap, **kwargs: Any) -> param.rx:  # noqa: ANN401
        """Create a Dotmap plot from a ManifoldMap object.

        This ensures the Dotmap is always linked to the
        current manifold map plot.

        Parameters
        ----------
        mm: ManifoldMap
            The ManifoldMap object to link the DotMap with.
        kwargs: dict
            Keyword arguments to pass to the DotMap operation.

        Returns
        -------
        rx: Reactive expression that updates when the ManifoldMap changes.

        """
        plot = param.rx(mm._plot_view)  # noqa: SLF001
        if mm.ls:
            plot.rx.watch(lambda _: mm.ls.param.update(selection_expr=None))
        return plot.rx.pipe(DotmapOp, **kwargs)


class DotmapOp(Operation):
    """Operation to generate a DotMap plot.

    Generates a DotMap plot from an existing HoloViews object
    backed by an AnnData object.
    """

    groupby = param.String(default="cell_type", doc="Column to group by.")

    ls = param.ClassSelector(class_=link_selections)

    marker_genes = param.Dict(default={}, doc="Dictionary of marker genes.")

    selection_expr = param.Parameter()

    def _apply(
        self,
        element: Overlay | Element,
        key: str | None = None,  # noqa: ARG002
    ) -> hv.Points:
        # We override ._apply instead of ._process to override the
        # tracking of the dataset and pipeline
        if isinstance(element, Overlay):
            element = element.get(0)
        if self.p.ls and self.p.ls.selection_expr:
            element = self.p.ls.filter(element.dataset)
        else:
            element = element.dataset
        return Dotmap(
            adata=element.data,
            marker_genes=self.p.marker_genes,
            groupby=self.p.groupby,
        ).plot()

    def __call__(
        self,
        element: DynamicMap | Overlay | Element,
        **kwargs: Any,  # noqa: ANN401
    ) -> DynamicMap | Overlay:
        """Apply the operation."""
        ls = kwargs.pop("ls", self.ls)
        if ls is None:
            return super().__call__(element, **kwargs)
        out = super().__call__(element, **kwargs)
        kwargs["streams"] = [Params(ls, ["selection_expr"])]
        inst = self.instance()
        return out.opts("Points", alpha=ls.unselected_alpha) * Operation.__call__(
            inst, element, ls=ls, **kwargs
        )
