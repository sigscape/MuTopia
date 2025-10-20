"""
Prebuilt track configurations for genome view plotting.

This module provides ready-to-use tracks built on top of the primitives in
``mutopia.plot.track_plot`` and the helpers in ``.transforms``.
"""

from __future__ import annotations
import numpy as np
from functools import partial
import mutopia.plot.track_plot as tr
from mutopia.palettes import categorical_palette, diverging_palette
from .transforms import TopographyTransformer, minmax_scale
from typing import Any, Callable, Optional, Sequence, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from mutopia.gtensor.gtensor import GTensorDataset
    from .track_plot import GenomeView

__all__ = [
    "plot_gene_annotation",
    "plot_marginal_observed_vs_expected",
    "plot_component_rates",
    "plot_topography",
    "plot_empirical_topography",
    "plot_gene_expression_track",
    "order_components",
]

def plot_gene_annotation(
    gtf: str,
    label: str = "Genes",
    all_labels_inside: bool = False,
    style: str = "flybase",
    label_genes: bool = True,
    fontsize: int = 5,
    ax_fn: Callable[["Axes"], Any] = lambda ax: ax.spines["bottom"].set_visible(False),
    **kw: Any,
) -> Callable[..., "Axes"]:
    """
    Create a gene annotation track from GTF file.

    Parameters
    ----------
    gtf : str
        Path to GTF file
    label : str, default "Genes"
        Track label
    all_labels_inside : bool, default False
        Whether to show all gene labels inside
    style : str, default "flybase"
        Gene track style
    fontsize : int, default 5
        Font size for labels
    ax_fn : callable
        Function to customize axes appearance
    **kw
        Additional keyword arguments

    Returns
    -------
    callable
        Track plotting function

    Examples
    --------
    >>> import mutopia.plot.track_plot as tr
    >>> cfg = lambda v: tr.tracks.gene_annotation_track("/path/genes.gtf")
    >>> _ = tr.plot_view(cfg, tr.make_view(ds, region="chr1:1-2_000_000"))
    """

    return tr.static_track(
        "GtfTrack",
        gtf,
        label=label,
        labels=label_genes,
        all_labels_inside=all_labels_inside,
        style=style,
        fontsize=fontsize,
        ax_fn=ax_fn,
        **kw,
    )


def plot_marginal_observed_vs_expected(
    view: "GenomeView",
    smooth: int = 20,
    pred_smooth: int = 10,
    label: str = "Mutation rate",
    legend: bool = True,
    height: float = 1,
    empirical_kw: Mapping[str, Any] = {"alpha": 0.5, "s": 0.1, "color": "lightgrey"},
    predicted_kw: Mapping[str, Any] = {
        "color": categorical_palette[1],
        "dashes": (1, 1),
        "alpha": 0.8,
        "linewidth": 0.75,
    },
    ax_fn: Callable[["Axes"], Any] = lambda ax: None,
) -> Callable[..., "Axes"]:
    """
    Compare empirical vs. predicted marginal mutation rates.

    Parameters
    ----------
    view : GenomeView
        Genome view that supplies smoothing and locus metadata.
    smooth : int, default 20
        Window size for smoothing empirical rates.
    pred_smooth : int, default 10
        Window size for smoothing predicted rates.
    label : str, default "Mutation rate"
        Track label.
    legend : bool, default True
        Whether to show a legend.
    height : float, default 1
        Track height.
    empirical_kw : dict
        Keyword args passed to the empirical scatterplot.
    predicted_kw : dict
        Keyword args passed to the predicted line plot.
    ax_fn : callable
        Optional axes customization function.

    Returns
    -------
    callable
        A stacked track containing empirical scatter and predicted line.

    Examples
    --------
    >>> import mutopia.plot.track_plot as tr
    >>> view = tr.make_view(ds, region="chr1:1_000_000-1_200_000")
    >>> cfg = lambda v: tr.tracks.marginal_observed_vs_expected(view)
    >>> _ = tr.plot_view(cfg, view)
    """
    return tr.stack_plots(
        tr.scatterplot(
            tr.pipeline(
                tr.select("empirical_marginal_locus"), view.smooth(smooth), tr.renorm
            ),
            **empirical_kw,
        ),
        tr.line_plot(
            tr.pipeline(
                tr.select("predicted_marginal_locus"),
                view.smooth(pred_smooth),
                tr.renorm,
            ),
            **predicted_kw,
        ),
        label=label,
        legend=legend,
        height=height,
        ax_fn=ax_fn,
    )


def plot_component_rates(
    view: "GenomeView",
    *components: Any,
    smooth: int = 30,
    height: float = 0.5,
    label: Optional[str] = None,
    color: str = categorical_palette[0],
    linewidth: float = 0.5,
) -> list[Callable[..., "Axes"]]:
    """
    Plot per-component mutation rates as filled line tracks.

    Parameters
    ----------
    view : GenomeView
        Genome view used for smoothing and renormalization.
    *components : Any
        Component identifiers to plot.
    smooth : int, default 30
        Smoothing window (in regions).
    height : float, default 0.5
        Track height per component.
    label : str, optional
        Label to use (defaults to the component identifier).
    color : str, default categorical_palette[0]
        Fill/line color.
    linewidth : float, default 0.5
        Line width.

    Returns
    -------
    list of callable
        One line_plot callable per component.
    """
    return [
        tr.line_plot(
            tr.pipeline(
                tr.select("component_distributions_locus", component=component),
                view.smooth(smooth),
                tr.renorm,
            ),
            fill=True,
            label=label or str(component),
            linewidth=linewidth,
            color=color,
            height=height,
        )
        for component in components
    ]


def _topography_ax_fn(ax: "Axes", transformer: TopographyTransformer):
    return (
        ax.grid(True, axis="y", linestyle="-", linewidth=0.4, color="white"),
        ax.set_yticks(np.arange(0, len(transformer.ordering_), 16), minor=False),
        ax.set_yticks(np.arange(8, len(transformer.ordering_), 16), minor=True),
        ax.set_yticklabels([]),
        ax.set_yticklabels(transformer.labels[::-1], minor=True, ha="right"),
        ax.tick_params(axis="y", which="minor", labelsize=7),
        ax.tick_params(axis="y", which="major", length=0),
    )


def plot_topography(
    transformer: TopographyTransformer,
    palette: str = "Greys",
    yticks: bool = False,
    label: str = "Predicted\ntopography",
    vmin: float = -3,
    vmax: float = 3,
    height: float = 1.5,
    **heatmap_kw: Any,
) -> Callable[..., "Axes"]:
    """
    Heatmap of predicted topography with hierarchical row order.

    Parameters
    ----------
    transformer : TopographyTransformer
        Fitted transformer that supplies transform, ordering, and labels.
    palette : str, default "Greys"
        Matplotlib colormap name.
    yticks : bool, default False
        Whether to render y tick labels.
    label : str, default "Predicted\\ntopography"
        Track label.
    vmin, vmax : float
        Color scale limits.
    height : float, default 1.5
        Track height.
    **heatmap_kw
        Extra kwargs forwarded to heatmap_plot.

    Returns
    -------
    callable
        Heatmap plotting callable.
    """
    return tr.heatmap_plot(
        transformer.transform,
        palette=palette,
        yticks=yticks,
        label=label,
        vmin=vmin,
        vmax=vmax,
        height=height,
        ax_fn=lambda ax: _topography_ax_fn(ax, transformer),
        **heatmap_kw,
    )


def plot_empirical_topography(
    transformer: TopographyTransformer,
    palette: str = "Greys",
    label: str = "Empirical\ntopography",
    height: float = 1.5,
    quantile_cutoff: float = 0.999,
    s: float = 0.01,
    alpha: float = 0.5,
    topography_kw: Mapping[str, Any] = {"vmin": -3, "vmax": 3, "cbar": False, "alpha": 0.15},
    **scatter_kw: Any,
) -> Callable[..., "Axes"]:
    """
    Overlay empirical topography points on a predicted topography heatmap.

    Parameters
    ----------
    transformer : TopographyTransformer
        Fitted transformer.
    palette : str, default "Greys"
        Colormap for the background heatmap.
    label : str, default "Empirical\\ntopography"
        Track label.
    height : float, default 1.5
        Track height.
    quantile_cutoff : float, default 0.999
        Upper quantile for clipping empirical intensities.
    s : float, default 0.01
        Scatter point size.
    alpha : float, default 0.5
        Scatter alpha.
    topography_kw : dict
        Extra kwargs passed to the background heatmap.
    **scatter_kw
        Extra kwargs passed to ax.scatter.

    Returns
    -------
    callable
        Stacked scatter + heatmap track.
    """
    from scipy.sparse import coo_matrix, csc_matrix

    def _get_heatmap(dataset: "GTensorDataset") -> coo_matrix:
        matrix = transformer._fetch_matrix("empirical_marginal", dataset)
        x = matrix[transformer.ordering_].values.T
        max_cut = np.quantile(x[np.isfinite(x)], quantile_cutoff)
        x = np.nan_to_num(x, nan=0.0, neginf=0.0)
        x = np.clip(x, a_min=0.0, a_max=max_cut)
        
        x = coo_matrix(x)
        x.eliminate_zeros()
        return x

    def _topography_scatter(
        ax: "Axes",
        *,
        dataset: "GTensorDataset",
        start: np.ndarray,
        end: np.ndarray,
        idx: np.ndarray,
        interval: tuple[int, int],
        **kw: Any,
    ) -> "Axes":
        matrix = _get_heatmap(dataset)
        matrix = coo_matrix(csc_matrix(matrix)[:, idx])  # unroll the matrix

        u = np.random.rand(len(matrix.data))
        x = ((end - start)[matrix.col]) * u + start[matrix.col]

        ax.scatter(x, matrix.row, c=matrix.data, cmap="Greys", **scatter_kw, s=s)
        ax.set_xlim(interval)

        return ax
    
    return tr.stack_plots(
        _topography_scatter,
        plot_topography(
            transformer,
            palette=palette,
            zorder=0,
            **topography_kw,
        ),
        label=label,
        height=height,
    )


def plot_gene_expression_track(
    expression_key: str = "GeneExpression",
    strand_key: str = "GeneStrand",
    linewidth: float = 0.5,
    label: str = "Gene\nexpression",
    color: str = "lightgrey",
    height: float = 1,
    log1p: bool = True,
) -> Callable[..., "Axes"]:
    """
    Strand-aware gene expression bar track (optionally symlog1p-transformed).

    Parameters
    ----------
    expression_key : str, default "GeneExpression"
        Dataset feature key for expression magnitude.
    strand_key : str, default "GeneStrand"
        Dataset feature key for strand (+1 / -1 / 0).
    linewidth : float, default 0.5
        Horizontal zero-line width.
    label : str, default "Gene\\nexpression"
        Track label.
    color : str, default "lightgrey"
        Bar color.
    height : float, default 1
        Track height.
    log1p : bool, default True
        If True, applies symmetric log1p to signed expression.

    Returns
    -------
    callable
        Bar plot callable.
    """
    import numpy as np

    return tr.bar_plot(
        tr.pipeline(
            tr.feature_matrix(expression_key, strand_key),
            lambda x: np.prod(x, axis=0),
            lambda x: np.sign(x) * np.log1p(np.abs(x)) if log1p else x,  # symlog1p
        ),
        ax_fn=lambda ax: (
            ax.axhline(
                0,
                color="k",
                linewidth=linewidth,
            ),
            ax.spines["bottom"].set_visible(False),
        ),
        label=label,
        color=color,
        height=height,
    )


def order_components(dataset: "GTensorDataset") -> np.ndarray:
    """
    Compute an ordering of components based on hierarchical clustering.

    Parameters
    ----------
    dataset : GTensorDataset
        Input dataset with component_distributions_locus.

    Returns
    -------
    ndarray
        Ordered component identifiers.
    """
    component_order = tr.pipeline(
        tr.select("component_distributions_locus"),
        lambda x : x.squeeze(),
        tr.apply_rows(tr.renorm),
        lambda x: x.to_pandas(),
        tr.reorder_df,
    )(dataset).index.values
    return component_order


if False:
    def component_rate_summary(
        view: "GenomeView",
        *,
        ideogram: str,
        scalebar_size: int = int(1e7),
        scalebar_scale: str = "mb",
        pred_smooth: int = 20,
        empirical_smooth: int = 10,
        legend: bool = True,
        pred_kw: Mapping[str, Any] = {
            "color": categorical_palette[1],
            "dashes": (1, 1),
            "alpha": 1,
            "linewidth": 0.75,
        },
        component_smooth: int = 30,
        component_order: Optional[Sequence[Any]] = None,
    ) -> tuple[Any, ...]:
        """
        Summary: scale bar, ideogram, observed vs predicted, and per-component rates.

        Parameters
        ----------
        view : GenomeView
            Genome view for smoothing, spacing, and region info.
        ideogram : str
            Path to cytoband file for ideogram.
        scalebar_size : int, default 1e7
            Scale bar size (bp).
        scalebar_scale : str, default "mb"
            Scale label units.
        pred_smooth : int, default 20
            Smoothing for predicted rate line.
        empirical_smooth : int, default 10
            Smoothing for empirical rate points.
        legend : bool, default True
            Whether to show legend in the rate plot.
        pred_kw : dict
            Predicted line kwargs.
        component_smooth : int, default 30
            Smoothing window for component rate tracks.
        component_order : sequence, optional
            Explicit component order; computed from dataset if None.

        Returns
        -------
        tuple
            A tuple of track callables consumable by tr.plot_view.
        """
        component_order = (
            order_components(view.dataset) if component_order is None else component_order
        )

        return (
            tr.scale_bar(scalebar_size, scale=scalebar_scale),
            tr.ideogram(ideogram, height=0.1),
            tr.tracks.plot_marginal_observed_vs_expected(
                view,
                smooth=empirical_smooth,
                pred_smooth=pred_smooth,
                predicted_kw=pred_kw,
                legend=legend,
            ),
            tr.spacer(0.1),
            *plot_component_rates(view, *component_order, smooth=component_smooth),
        )



    def mega_summary(
        view,
        *,
        dataset,
        mutation_rate_plot_kw={},
        shap_plot_kw={},
    ):

        import seaborn as sns
        from mutopia.plot import plot_spectrum, plot_shap_summary
        from mutopia.gtensor import fetch_component

        feature_order = [
            "GeneExpression",
            "H3K36me3",
            "POLR2A",
            "H3K27me3",
            "H3K9me3",
            "H3K4me1",
            "H3K4me3",
            "H3K27ac",
            "ATACAccessible",
            "DNase",
            "HICEigenvector",
        ]

        def _plot_shared_effects(component_name, ax):
            shared_effect = (
                dataset.sections["Spectra"]["shared_effects"]
                .sel(component=component_name)
                .to_pandas()
                .iloc[1:,]
                .to_frame()
                .T
            )

            sns.heatmap(
                shared_effect,
                cmap=diverging_palette,
                linewidths=0.5,
                vmin=-0.5,
                vmax=0.5,
                cbar=False,
                ax=ax,
                square=True,
            )
            ax.set(
                ylabel="",
                xlabel="",
                xticklabels=[],
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("lightgrey")
                spine.set_linewidth(0.5)

            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        def _plot_component(comp_name, ax):
            ax = plot_spectrum(fetch_component(dataset, comp_name), ax=ax)
            ax.set_ylabel(
                comp_name, fontsize=9, rotation=0, labelpad=10, va="center", ha="right"
            )
            return ax

        def _plot_shap(comp_name, ax):
            defaults = dict(
                cbar=False,
                scale=50,
                max_size=3,
                alpha=0.8,
            )
            defaults.update(shap_plot_kw)

            ax = plot_shap_summary(
                dataset,
                component_order=[comp_name],
                feature_order=feature_order + ["Repliseq" + phase for phase in phases],
                ax=ax,
                **defaults,
            )
            ax.set(
                ylabel="",
                xlabel="",
                yticklabels=[],
                xticklabels=[],
            )
            return ax

        component_order = tr.order_components(dataset)

        phases = ["G1b", "S1", "S2", "S3", "S4", "G2"]

        feature_hm = tr.heatmap_plot(
            tr.pipeline(
                tr.feature_matrix(*feature_order),
                tr.clip(0, 0.97),
                lambda x: np.log1p(x),
                tr.apply_rows(minmax_scale),
                lambda x: x.fillna(0.0),
            ),
            palette="viridis",
            cbar=False,
            label="Functional\nfeatures",
        )

        repliseq = tr.heatmap_plot(
            tr.pipeline(
                tr.feature_matrix(*["Repliseq" + phase for phase in phases]),
                tr.apply_rows(minmax_scale),
            ),
            palette="crest_r",
            cbar=False,
            label="Cell cycle\nphase",
            ax_fn=lambda ax: (ax.set_yticklabels(reversed(phases))),
        )

        return (
            tr.columns(
                ...,
                tr.scale_bar(1_000_000, scale="mb"),
                ...,
                ...,
                height=0.1,
            ),
            tr.columns(
                ...,
                tr.ideogram(
                    "/Users/allen/projects/mutopia/signaturemodels/notebooks/cytoBand.txt",
                    height=0.1,
                ),
                ...,
                ...,
                height=0.1,
            ),
            tr.columns(
                ...,
                tr.tracks.marginal_observed_vs_expected(
                    view,
                    legend=False,
                    label="Mutation\nrate",
                    **mutation_rate_plot_kw,
                ),
                ...,
                ...,
            ),
            tr.columns(
                ...,
                feature_hm,
                ...,
                ...,
                height=1.5,
            ),
            tr.columns(
                ...,
                repliseq,
                ...,
                ...,
                height=0.95,
            ),
            tr.columns(
                tr.text_banner("Component spectra"),
                tr.text_banner("Component mutation rates"),
                tr.text_banner("Strand effects"),
                tr.text_banner("SHAP value summaries"),
                height=0.5,
            ),
            [
                tr.columns(
                    tr.custom_plot(partial(_plot_component, component_name)),
                    tr.component_rates(view, component_name, label=" ", smooth=20)[0],
                    tr.custom_plot(partial(_plot_shared_effects, component_name)),
                    tr.custom_plot(partial(_plot_shap, component_name)),
                    height=0.5,
                )
                for component_name in component_order
            ],
        )


    def plot_mega_summary(dataset, **kw):

        view = tr.make_view(dataset, "chr1:49007569-69616441", title=None)

        return tr.plot_view(
            mega_summary,
            view,
            dataset=dataset,
            **kw,
            width_ratios=[3, 4, 1, 4],
            width=11,
            gridpsec_kw={"hspace": 0.3, "wspace": 0.2},
        )
