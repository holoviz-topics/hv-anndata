"""HoloViz plotting using AnnData as input."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import chain
from typing import TYPE_CHECKING, TypedDict

import anndata as ad
import holoviews as hv
import pandas as pd
import panel as pn
import panel_material_ui as pmui
import param
from bokeh.models import CustomJSTickFormatter
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


class _CreateDotmapPlotParams(TypedDict):
    kdims: NotRequired[list[str | hv.Dimension]]
    vdims: NotRequired[list[str | hv.Dimension]]
    marker_genes: dict[str, list[str]] | list[str]
    groupby: str
    expression_cutoff: NotRequired[float]
    max_dot_size: NotRequired[int]
    standard_scale: NotRequired[Literal["var", "group"] | None]
    use_raw: NotRequired[bool | None]
    mean_only_expressed: NotRequired[bool]


class _DotmapPlotParams(_CreateDotmapPlotParams):
    adata: ad.AnnData
    ls: link_selections


_DEFAULT_GROUPBY = "cell_type"
_DEFAULT_KDIMS = ["marker_line", "cluster"]
_DEFAULT_VDIMS = [
    "gene_id",
    "mean_expression",
    "percentage",
    "marker_cluster_name",
]
_DEFAULT_USE_RAW = None
_DEFAULT_EXPRESSION_CUTOFF = 0.0
_DEFAULT_MEAN_ONLY_EXPRESSED = False
_DEFAULT_MAX_DOT_SIZE = 20
_DEFAULT_STANDARD_SCALE = None


def _prepare_data(  # noqa: C901, PLR0912, PLR0913, PLR0915
    adata: ad.AnnData,
    *,
    groupby: str = _DEFAULT_GROUPBY,
    kdims: list[str | hv.Dimension] = _DEFAULT_KDIMS,
    vdims: list[str | hv.Dimension] = _DEFAULT_VDIMS,
    marker_genes: dict[str, list[str]] | list[str] | None = None,
    use_raw: bool | None = _DEFAULT_USE_RAW,
    expression_cutoff: float = _DEFAULT_EXPRESSION_CUTOFF,
    mean_only_expressed: bool = _DEFAULT_MEAN_ONLY_EXPRESSED,
    standard_scale: Literal["var", "group"] | None = _DEFAULT_STANDARD_SCALE,
) -> pd.DataFrame:
    if marker_genes is None:
        marker_genes = {}
    # Flatten the marker_genes preserving order
    is_mapping_marker_genes = isinstance(marker_genes, Mapping)
    if is_mapping_marker_genes:
        all_marker_genes = list(
            dict.fromkeys(chain.from_iterable(marker_genes.values()))
        )
    else:
        all_marker_genes = list(marker_genes)

    # Determine to use raw or processed
    if use_raw is None:
        use_raw = adata.raw is not None
    elif use_raw and adata.raw is None:
        err = "use_raw=True but .raw attribute is not present in adata"
        raise ValueError(err)

    # Check which genes are actually present in the correct location
    if use_raw and adata.raw is not None:
        available_var_names = adata.raw.var_names
    else:
        available_var_names = adata.var_names

    missing_genes = set(all_marker_genes) - set(available_var_names)
    if missing_genes:
        print(  # noqa: T201
            f"Warning: The following genes are not present in the dataset and will be skipped: {missing_genes}"  # noqa: E501
        )
        available_marker_genes = [g for g in all_marker_genes if g not in missing_genes]
        if not available_marker_genes:
            msg = "None of the specified marker genes are present in the dataset."
            raise ValueError(msg)
    else:
        available_marker_genes = all_marker_genes

    # Subset the data with only available genes
    if use_raw and adata.raw is not None:
        adata_subset = adata.raw[:, available_marker_genes]
        expression_df = pd.DataFrame(
            adata_subset.X.toarray()
            if hasattr(adata_subset.X, "toarray")
            else adata_subset.X,
            index=adata.obs_names,
            columns=available_marker_genes,
        )
    else:
        expression_df = adata[:, available_marker_genes].to_df()

    # Join with groupby column
    joined_df = expression_df.join(adata.obs[groupby])

    def compute_expression(df: pd.DataFrame) -> pd.DataFrame:
        # Separate the groupby column from gene columns
        gene_cols = [col for col in df.columns if col != groupby]

        results = {}
        for gene in gene_cols:
            gene_data = df[gene]

            # percentage of expressing cells
            percentage = (gene_data > expression_cutoff).mean() * 100

            if mean_only_expressed:
                expressing_mask = gene_data > expression_cutoff
                if expressing_mask.any():
                    mean_expr = gene_data[expressing_mask].mean()
                else:
                    mean_expr = 0.0
            else:
                mean_expr = gene_data.mean()

            results[gene] = {"percentage": percentage, "mean_expression": mean_expr}

        return pd.DataFrame(results).T

    grouped = joined_df.groupby(groupby, observed=True)
    expression_stats = grouped.apply(compute_expression, include_groups=False)

    if is_mapping_marker_genes:
        data = [  # Likely faster way to do this, but harder to read
            expression_stats.xs(gene, level=1)
            .reset_index(names="cluster")
            .assign(
                marker_cluster_name=marker_cluster_name,
                gene_id=gene,
            )
            for marker_cluster_name, gene_list in marker_genes.items()
            for gene in gene_list
            # Only include genes that weren't filtered out
            if gene in available_marker_genes
        ]
    else:
        data = [  # Likely faster way to do this, but harder to read
            expression_stats.xs(gene, level=1)
            .reset_index(names="cluster")
            .assign(gene_id=gene)
            for gene in marker_genes
            # Only include genes that weren't filtered out
            if gene in available_marker_genes
        ]

    if data:
        df = pd.concat(data, ignore_index=True)
    else:
        df = pd.DataFrame({k: [] for k in kdims + vdims})

    # Apply standard_scale if specified
    if standard_scale == "var":
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

    elif standard_scale == "group":
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


def _get_opts(
    *,
    kdims: list[str | hv.Dimension] = _DEFAULT_KDIMS,
    vdims: list[str | hv.Dimension] = _DEFAULT_VDIMS,
    marker_genes: dict[str, list[str]] | list[str] | None = None,
    max_dot_size: int = _DEFAULT_MAX_DOT_SIZE,
    plot_opts: Mapping[str, Any],
) -> dict[str, Any]:
    opts = dict(
        cmap="Reds",
        color=hv.dim("mean_expression"),
        colorbar=True,
        show_legend=False,
        xrotation=45,
    )

    radius_dim = hv.dim("percentage")
    match hv.Store.current_backend:
        case "matplotlib":
            backend_opts = {"s": radius_dim * max_dot_size}
        case "bokeh":
            hover_tooltips = [*kdims, *vdims]
            if "marker_cluster_name" in hover_tooltips and (
                not isinstance(marker_genes, Mapping)
            ):
                hover_tooltips.remove("marker_cluster_name")
            backend_opts = {
                "colorbar_position": "left",
                "clabel": "Mean Expression",
                "colorbar_opts": {
                    # Does not seem to work
                    "title_text_font_style": "italic",
                },
                "line_alpha": 0.2,
                "line_color": "k",
                "tools": ["hover"],
                "hover_tooltips": hover_tooltips,
                "height": 500,
                "responsive": "width",
                # Saw layout issues with the dendrogram
                # "min_height": 300,  # noqa: ERA001
                # "responsive": True,  # noqa: ERA001
            }
            if _HOLOVIEWS_VERSION >= (1, 21, 0):
                backend_opts |= {"radius": radius_dim / 100 / 2}
                if _HOLOVIEWS_VERSION >= (1, 22, 0):
                    sb_formatter = CustomJSTickFormatter(
                        code="""
                        return Math.round(tick * 2 * 100, 2) + "%";
                        """
                    )
                    backend_opts |= {
                        "sizebar": True,
                        "sizebar_location": "left",
                        "sizebar_orientation": "vertical",
                        "sizebar_opts": {
                            "title": "Fraction of\ncells (%)",
                            "title_standoff": 15,
                            "formatter": sb_formatter,
                        },
                    }
            else:
                backend_opts |= {"size": radius_dim.norm() * max_dot_size}
        case _:
            backend_opts = {}

    return opts | backend_opts | plot_opts


def _get_cat_obs(adata: ad.AnnData) -> list[str]:
    categorical_obs = adata.obs.select_dtypes(include=["category"]).columns.tolist()
    return sorted(categorical_obs)


def create_dotmap_plot(  # noqa: PLR0913
    adata: ad.AnnData,
    *,
    groupby: str | None = _DEFAULT_GROUPBY,
    kdims: list[str | hv.Dimension] = _DEFAULT_KDIMS,
    vdims: list[str | hv.Dimension] = _DEFAULT_VDIMS,
    marker_genes: Mapping[str, list[str]] | list[str] | None = None,
    use_raw: bool | None = _DEFAULT_USE_RAW,
    expression_cutoff: float = _DEFAULT_EXPRESSION_CUTOFF,
    mean_only_expressed: bool = _DEFAULT_MEAN_ONLY_EXPRESSED,
    standard_scale: Literal["var", "group"] | None = _DEFAULT_STANDARD_SCALE,
    max_dot_size: int = 20,
    plot_opts: Mapping[str, Any] | None = None,
) -> hv.Element:
    """Create a Dotmap plot from an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    groupby : str | None, optional
        Observation column to group by, by default "cell_type". If None,
        first categorical obs of adata.
    kdims : list[str  |  hv.Dimension], optional
        Key dimensions representing cluster and marker line
        (combined marker cluster name and gene), by default
        ``["marker_line", "cluster"]``
    vdims : list[str  |  hv.Dimension], optional
        Value dimensions representing expression metrics and metadata
    marker_genes : Mapping[str, list[str]] | list[str] | None, optional
        Dictionary or list of marker genes, by default None
    use_raw : bool | None, optional
        Whether to use `.raw` attribute of AnnData, by default None
    expression_cutoff : float, optional
        Cutoff for expression, by default 0.0
    mean_only_expressed : bool, optional
        If True, gene expression is averaged only over expressing cells,
        by default False
    standard_scale : Literal["var";, "group", None], optional
        Whether to standardize the dimension between 0 and 1. 'var' scales each
        gene, 'group' scales each cell type, by default None
    max_dot_size : int, optional
        Maximum size of the dots, by default 20
    plot_opts: dict, optional
        HoloViews plot options for the Points element.

    Returns
    -------
    hv.Points
        Dotmap plot instance.

    """
    if groupby is None:
        groupby = _get_cat_obs(adata)[0]
    data = _prepare_data(
        adata,
        groupby=groupby,
        kdims=kdims,
        vdims=vdims,
        marker_genes=marker_genes,
        use_raw=use_raw,
        expression_cutoff=expression_cutoff,
        mean_only_expressed=mean_only_expressed,
        standard_scale=standard_scale,
    )
    plot = hv.Points(data, kdims=kdims, vdims=vdims)
    plot_opts = plot_opts if plot_opts else {}
    opts = _get_opts(
        kdims=kdims,
        vdims=vdims,
        marker_genes=marker_genes,
        max_dot_size=max_dot_size,
        plot_opts=plot_opts,
    )
    return plot.opts(**opts)


class DotmapParams(param.Parameterized):
    """Shared parameters."""

    kdims = param.List(
        default=_DEFAULT_KDIMS,
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
    marker_genes = param.ClassSelector(
        default={}, class_=(dict, list), doc="Dictionary or list of marker genes."
    )
    expression_cutoff = param.Number(
        default=_DEFAULT_EXPRESSION_CUTOFF, doc="Cutoff for expression."
    )
    max_dot_size = param.Integer(
        default=_DEFAULT_MAX_DOT_SIZE, doc="Maximum size of the dots."
    )

    standard_scale = param.Selector(
        default=_DEFAULT_STANDARD_SCALE,
        objects=[None, "var", "group"],
        doc="""\
        Whether to standardize the dimension between 0 and 1. 'var' scales each gene,
        'group' scales each cell type.""",
    )

    use_raw = param.Boolean(
        default=_DEFAULT_USE_RAW,
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
        default=_DEFAULT_MEAN_ONLY_EXPRESSED,
        doc="If True, gene expression is averaged only over expressing cells.",
    )

    plot_opts = param.Dict(doc="HoloViews plot options for the Points element.")


class Dotmap(pn.viewable.Viewer, DotmapParams):
    """Create a DotmapPlot from anndata."""

    adata = param.ClassSelector(class_=ad.AnnData)

    groupby = param.String(default=_DEFAULT_GROUPBY, allow_None=True)

    dendrogram = param.ClassSelector(
        default=None,
        class_=(type(None), bool, hv.operation.dendrogram),
        doc="Add a dendrogram.",
    )

    dendrogram_opts = param.Dict(
        doc="HoloViews plot options for the Dendrogram element."
    )

    ls = param.ClassSelector(class_=link_selections)

    _input_dm = param.ClassSelector(default=None, allow_None=True, class_=hv.DynamicMap)

    def __init__(self, **params: Unpack[_DotmapPlotParams]) -> None:
        """Init the Dotmap."""
        required = {"adata", "marker_genes"}
        if "_input_dm" in params:
            required.remove("adata")
        if missing := (required - params.keys()):
            msg = f"Needs to have the following argument(s): {missing}"
            raise TypeError(msg)
        self._w_gb = pmui.widgets.Select(
            options=["bar", "foo"],
            label="Group by",
            sizing_mode="stretch_width",
        )
        self._w_gb.param.watch(lambda e: self.param.update(groupby=e.new), ["value"])
        super().__init__(**params)
        self._w_gs = GeneSelector(label="Marker Genes:", value=self.param.marker_genes)
        self._w_gs.param.watch(self._update_marker_genes, "value", onlychanged=False)

    @param.depends("adata", "_input_dm", watch=True, on_init=True)
    def _on_input_data(self) -> None:
        if self.adata is None:
            obj = self._input_dm
            if isinstance(obj, hv.DynamicMap):
                obj = obj[()]
            if isinstance(obj, Overlay):
                obj = obj.get(0)
            self.adata = obj.dataset.data
        groupby_opts = _get_cat_obs(self.adata)
        self._w_gb.options = groupby_opts
        self.groupby = groupby_opts[0]

    @param.depends("groupby", watch=True, on_init=True)
    def _on_groupby(self) -> None:
        if not self._w_gb.options:
            return
        if self.groupby not in self._w_gb.options:
            groupby = self._w_gb.options[0]
            with param.parameterized.discard_events(self):
                self.groupby = groupby
        self._w_gb.value = self.groupby

    def _get_dendrogram(self) -> hv.operation.dendrogram | None:
        if isinstance(self.dendrogram, hv.operation.dendrogram):
            return self.dendrogram
        if self.dendrogram:
            return hv.operation.dendrogram.instance(
                adjoint_dims=["cluster"],
                main_dim="mean_expression",
                invert=True,
            )
        return None

    @param.depends("marker_genes", "groupby")
    def plot(self) -> hv.Points:
        """Plot the Dotmap."""
        ignored = ["_input_dm", "adata", "dendrogram", "dendrogram_opts", "name"]
        kwargs = {k: v for k, v in self.param.values().items() if k not in ignored}
        if self._input_dm is not None:
            obj = self._input_dm
        else:
            # Dummy dataset to pass it to the operation which needs the adata object
            obj = hv.Dataset([])
            obj.data = self.adata

        out = DotmapOp(obj, **kwargs)
        if self.dendrogram is not None:
            dd = self._get_dendrogram()
            dd_opts = self.dendrogram_opts or {}
            out = dd(out).opts(**dd_opts)
        return out

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
        widgets = pmui.Column(
            self._w_gb,
            self._w_gs,
            sizing_mode="stretch_width",
            max_width=self._w_gs.width,
            sx={"border": 1, "borderColor": "#e3e3e3", "borderRadius": 1},
        )
        return pn.Row(widgets, self.plot)


def dotmap_from_manifoldmap(
    mm: ManifoldMap,
    **kwargs: Unpack[_CreateDotmapPlotParams],
) -> param.rx:
    """Create a Dotmap from a ManifoldMap object.

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
    mm_plot = param.rx(mm._plot_view)  # noqa: SLF001
    if mm.ls:
        mm_plot.rx.watch(lambda _: mm.ls.param.update(selection_expr=None))

    def on_mm_updates(new_element: hv.DynamicMap) -> Dotmap:
        return Dotmap(_input_dm=new_element, ls=mm.ls, **kwargs)

    return mm_plot.rx.pipe(on_mm_updates)


class DotmapOp(Operation, DotmapParams):
    """Operation to generate a DotMap plot.

    Generates a DotMap plot from an existing HoloViews object
    backed by an AnnData object.
    """

    groupby = param.String(
        default=_DEFAULT_GROUPBY, doc="Observation column to group by"
    )

    ls = param.ClassSelector(class_=link_selections)

    def _apply(
        self,
        element: Overlay | Element,
        key: str | None = None,  # noqa: ARG002
    ) -> hv.Points:
        # We override ._apply instead of ._process to override the
        # tracking of the dataset and pipeline
        if isinstance(element, hv.DynamicMap):
            element = element.data[()]
        if isinstance(element, Overlay):
            element = element.get(0)
        if self.p.ls and self.p.ls.selection_expr:
            element = self.p.ls.filter(element.dataset)
        else:
            element = element.dataset
        return create_dotmap_plot(
            adata=element.data,
            groupby=self.p.groupby,
            **{k: v for k, v in self.p.items() if k in DotmapParams.param},
        )

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
        return out.opts(
            "Points", alpha=ls.unselected_alpha, sizebar=False
        ) * Operation.__call__(inst, element, ls=ls, **kwargs)
