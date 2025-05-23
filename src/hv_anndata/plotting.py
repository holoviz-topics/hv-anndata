"""DotmapPlot using AnnData as input and return a holoviews plot."""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Any, TypedDict

import anndata as ad
import holoviews as hv
import pandas as pd
import param

if TYPE_CHECKING:
    from typing import NotRequired, Unpack


class _DotmatPlotParams(TypedDict):
    kdims: NotRequired[list[str | hv.Dimension]]
    vdims: NotRequired[list[str | hv.Dimension]]
    adata: ad.AnnData
    marker_genes: dict[str, list[str]]
    groupby: str
    expression_cutoff: NotRequired[float]
    max_dot_size: NotRequired[int]
    standard_scale: NotRequired[str | None]
    use_raw: NotRequired[bool | None]
    mean_only_expressed: NotRequired[bool]


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
        ],
        doc="Value dimensions representing expression metrics and metadata.",
    )

    adata = param.ClassSelector(class_=ad.AnnData)
    marker_genes = param.Dict(default={}, doc="Dictionary of marker genes.")
    groupby = param.String(default="cell_type", doc="Column to group by.")
    expression_cutoff = param.Number(default=0.0, doc="Cutoff for expression.")
    max_dot_size = param.Integer(default=20, doc="Maximum size of the dots.")

    standard_scale = param.Selector(
        default=None,
        objects=[None, "var", "group"],
        doc="Whether to standardize the dimension between 0 and 1. 'var' scales each gene, 'group' scales each cell type.",
    )

    use_raw = param.Selector(
        default=None,
        objects=[None, True, False],
        doc="Use `.raw` attribute of AnnData if present. If None, uses .raw if present.",
    )

    mean_only_expressed = param.Boolean(
        default=False,
        doc="If True, gene expression is averaged only over expressing cells.",
    )

    def _prepare_data(self) -> pd.DataFrame:
        # Flatten the marker_genes preserving order and duplicates
        all_marker_genes = list(chain.from_iterable(self.p.marker_genes.values()))

        # Determine to use raw or processed
        use_raw = self.p.use_raw
        if use_raw is None:
            use_raw = self.p.adata.raw is not None
        if use_raw and self.p.adata.raw is not None:
            adata_subset = self.p.adata.raw[:, all_marker_genes]
            expression_df = pd.DataFrame(
                adata_subset.X.toarray()
                if hasattr(adata_subset.X, "toarray")
                else adata_subset.X,
                index=self.p.adata.obs_names,
                columns=all_marker_genes,
            )
        else:
            adata_subset = self.p.adata[:, all_marker_genes]
            expression_df = pd.DataFrame(
                adata_subset.X.toarray()
                if hasattr(adata_subset.X, "toarray")
                else adata_subset.X,
                index=self.p.adata.obs_names,
                columns=all_marker_genes,
            )

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

        # expression data for the included marker genes
        joined_df = expression_df.join(self.p.adata.obs[self.p.groupby])

        def compute_expression(df: pd.DataFrame) -> pd.DataFrame:
            # Separate the groupby column from gene columns
            gene_cols = [col for col in df.columns if col != self.p.groupby]

            results = {}
            for gene in gene_cols:
                gene_data = df[gene]

                # percentage of expressing cells
                percentage = (gene_data > self.p.expression_cutoff).mean() * 100

                if self.p.mean_only_expressed:
                    expressing_mask = gene_data > self.p.expression_cutoff
                    if expressing_mask.any():
                        mean_expr = gene_data[expressing_mask].mean()
                    else:
                        mean_expr = 0.0
                else:
                    mean_expr = gene_data.mean()

                results[gene] = pd.Series(
                    {"percentage": percentage, "mean_expression": mean_expr}
                )

            return pd.DataFrame(results).T

        grouped = joined_df.groupby(self.p.groupby, observed=True)
        expression_stats = grouped.apply(compute_expression, include_groups=False)

        data = []
        for marker_cluster_name, gene_list in self.p.marker_genes.items():
            for gene in gene_list:
                if gene in all_marker_genes:
                    gene_stats = expression_stats.xs(gene, level=1)
                    for cluster in gene_stats.index:
                        data.append(
                            {
                                "cluster": cluster,
                                "gene_id": gene,
                                "marker_cluster_name": marker_cluster_name,
                                "percentage": gene_stats.loc[cluster, "percentage"],
                                "mean_expression": gene_stats.loc[
                                    cluster, "mean_expression"
                                ],
                            }
                        )

        df = pd.DataFrame(data)

        # Apply standard_scale if specified
        if self.p.standard_scale == "var":
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

        elif self.p.standard_scale == "group":
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
        df["marker_line"] = df["marker_cluster_name"] + ", " + df["gene_id"]

        return df

    def _get_opts(self) -> dict[str, Any]:
        opts = dict(
            cmap="Reds",
            color=hv.dim("mean_expression"),  # Better if we could avoid this one
            colorbar=True,
            show_legend=False,
            xrotation=45,
            line_alpha=0.2,
            line_color="k",
        )
        size_dim = hv.dim("percentage").norm() * self.p.max_dot_size
        match hv.Store.current_backend:
            case "matplotlib":
                backend_opts = {"s": size_dim}
            case "bokeh":
                backend_opts = {
                    "size": size_dim,
                    "colorbar_position": "left",
                    "tools": ["hover"],
                    "width": 900,
                    "height": 500,
                }
            case _:
                backend_opts = {}

        return opts | backend_opts

    def __call__(self, **params: Unpack[_DotmatPlotParams]) -> hv.Points:
        """Create a DotmapPlot from anndata."""
        if required := {"adata", "marker_genes", "groupby"} - params.keys():
            msg = f"Needs to have the following argument(s): {required}"
            raise TypeError(msg)
        self.p = param.ParamOverrides(self, params)

        df = self._prepare_data()  # noqa: PD901
        plot = hv.Points(df, kdims=self.p.kdims, vdims=self.p.vdims, group="dotmap")
        plot.opts(**self._get_opts())
        return plot
