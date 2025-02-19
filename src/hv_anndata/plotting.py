"""DotmapPlot using AnnData as input and return a holoviews plot."""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, TypedDict

import anndata as ad
import holoviews as hv
import pandas as pd
import param

if TYPE_CHECKING:
    from typing_extensions import NotRequired

hv.extension("bokeh")


class _DotmatPlotParams(TypedDict):
    kdims: NotRequired[list[str | hv.Dimension]]
    vdims: NotRequired[list[str | hv.Dimension]]
    adata: ad.AnnData
    marker_genes: dict[str, list[str]]
    groupby: str
    expression_cutoff: NotRequired[float]
    max_dot_size: NotRequired[int]


class Dotmap(param.ParameterizedFunction):
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
            "mean_expression_norm",
        ],
        doc="Value dimensions representing expression metrics and metadata.",
    )

    adata = param.ClassSelector(class_=ad.AnnData)
    marker_genes = param.Dict(default={}, doc="Dictionary of marker genes.")
    groupby = param.String(default="cell_type", doc="Column to group by.")
    expression_cutoff = param.Number(default=0.1, doc="Cutoff for expression.")

    max_dot_size = param.Integer(default=15, doc="Maximum size of the dots.")

    def _prepare_data(self) -> pd.DataFrame:
        # Flatten the marker_genes preserving order and duplicates
        all_marker_genes = list(chain.from_iterable(self.p.marker_genes.values()))

        # Check if all genes are present in adata.var_names, warn about missing ones
        missing_genes = set(all_marker_genes) - set(self.p.adata.var_names)
        if missing_genes:
            print(  # noqa: T201
                f"Warning: The following genes are not present in the dataset and will be skipped: {missing_genes}"  # noqa: E501
            )
            all_marker_genes = [g for g in all_marker_genes if g not in missing_genes]
            if not all_marker_genes:
                msg = "None of the specified marker genes are present in the dataset."
                raise ValueError(msg)

        # Extract expression data for the included marker genes
        expression_df = self.p.adata[:, all_marker_genes].to_df()
        joined_df = expression_df.join(self.p.adata.obs[self.p.groupby])

        def compute_expression(df: pd.DataFrame) -> pd.DataFrame:
            percentages = (df > self.p.expression_cutoff).mean() * 100
            mean_expressions = df.mean()
            return pd.DataFrame(
                {"percentage": percentages, "mean_expression": mean_expressions}
            )

        grouped = joined_df.groupby(self.p.groupby, observed=True)
        expression_stats = grouped.apply(
            compute_expression, include_groups=False
        ).drop_duplicates()

        # Likely faster way to do this, but harder to read
        data = [
            expression_stats.xs(gene, level=1)
            .reset_index(names="cluster")
            .assign(
                marker_cluster_name=marker_cluster_name,
                gene_id=gene,
            )
            for marker_cluster_name, gene_list in self.p.marker_genes.items()
            for gene in gene_list
        ]
        df = pd.concat(data, ignore_index=True)  # noqa: PD901
        df["marker_line"] = df["marker_cluster_name"] + ", " + df["gene_id"]
        df["mean_expression_norm"] = df.groupby("marker_line")[
            "mean_expression"
        ].transform(lambda x: x / xmax if (xmax := x.max()) > 0 else 0)
        return df

    def __call__(self, **params: _DotmatPlotParams) -> hv.Points:
        """Create a DotmapPlot from anndata."""
        if required := {"adata", "marker_genes", "groupby"} - params.keys():
            msg = f"Needs to have the following argument(s): {required}"
            raise ValueError(msg)
        self.p = param.ParamOverrides(self, params)

        df = self._prepare_data()  # noqa: PD901
        plot = hv.Points(df, kdims=self.p.kdims, vdims=self.p.vdims, group="dotmap")
        plot.opts(
            # Would be better if we could avoid this one
            color=hv.dim("mean_expression_norm"),
            size=hv.dim("percentage").norm() * self.p.max_dot_size,
        )
        return plot


hv.opts.defaults(
    hv.opts.Points(
        "dotmap",
        cmap="Reds",
        responsive=True,
        min_height=380,
        hover_tooltips=["marker_line", "cluster", "mean_expression", "percentage"],
        ylabel="Cluster",
        xlabel="Marker Cluster, Gene",
        xrotation=45,
        colorbar=True,
        colorbar_position="left",
        invert_yaxis=True,
        show_legend=False,
        fontscale=0.7,
        xaxis="top",
        clabel="Mean expression in group",
    )
)
