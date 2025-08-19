import numpy as np
import mutopia.plot.track_plot as tr
from mutopia.utils import categorical_palette
from .track_plot import static_track
from .transforms import TopographyTransformer

def gene_annotation_track(
    gtf,
    label="Genes",
    all_labels_inside=False,
    style="flybase",
    fontsize=5,
    ax_fn=lambda ax: ax.spines["bottom"].set_visible(False),
    **kw,
):
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
    """

    return static_track(
        "GtfTrack",
        gtf,
        label=label,
        all_labels_inside=all_labels_inside,
        style=style,
        fontsize=fontsize,
        ax_fn=ax_fn,
        **kw,
    )


def marginal_observed_vs_expected(
    view,
    smooth=30,
    pred_smooth=10,
    label="Mutation rate",
    legend=True,
    height=1,
    empirical_kw={"alpha": 0.5, "s": 0.1, "color": "lightgrey"},
    predicted_kw={"color": categorical_palette[1], "dashes": (1, 1), "alpha": 0.5, "linewidth": 0.75},
):
    return tr.stack_plots(
        tr.scatterplot(
            tr.pipeline(tr.select("empirical_marginal_locus"), view.smooth(smooth), tr.renorm),
            **empirical_kw
        ),
        tr.line_plot(
            tr.pipeline(tr.select("predicted_marginal_locus"), view.smooth(pred_smooth), tr.renorm),
            **predicted_kw,
        ),
        label=label,
        legend=legend,
        height=height,
    )


def component_rates(
    view,
    *components,
    smooth=30,
    height=0.5,
    label=None,
    color=categorical_palette[0],
    linewidth=0.5,
):
    return [
        tr.line_plot(
            tr.pipeline(
                tr.select("component_distributions_locus", component=component),
                view.smooth(smooth),
                tr.renorm
            ),
            fill=True,
            label=label or str(component),
            linewidth=linewidth,
            color=color,
            height=height,
        )
        for component in components
    ]


def _topography_ax_fn(ax, transformer: TopographyTransformer):
    return (
        ax.grid(True, axis='y', linestyle='-', linewidth=0.4, color="white"),
        ax.set_yticks(np.arange(0, len(transformer.ordering_), 16), minor=False),
        ax.set_yticks(np.arange(8, len(transformer.ordering_), 16), minor=True),
        ax.set_yticklabels([]),
        ax.set_yticklabels(transformer.labels[::-1], minor=True, ha="right"),
        ax.tick_params(axis='y', which='minor', labelsize=7),
        ax.tick_params(axis='y', which='major', length=0)
    )


def topography(
    transformer : TopographyTransformer,
    palette="Greys",
    yticks=False,
    label='Predicted\ntopography',
    vmin=-3, 
    vmax=3,
    height=1.5,
    **heatmap_kw,
):
    return tr.heatmap_plot(
        transformer.transform,
        palette=palette,
        yticks=yticks,
        label=label,
        vmin=vmin, 
        vmax=vmax,
        height=height,
        ax_fn = lambda ax: _topography_ax_fn(ax, transformer),
        **heatmap_kw,
    )


def empirical_topography(
    transformer : TopographyTransformer,
    palette="Greys",
    label='Empirical\ntopography',
    height=1.5,
    quantile_cutoff=0.999,
    topography_kw = {"vmin": -3, "vmax": 3, "cbar": False},
    **heatmap_kw,
):
    def _get_heatmap(dataset):
        matrix = transformer._fetch_matrix("empirical_marginal", dataset)
        x = matrix[transformer.ordering_].values.T
        return np.clip(x, a_min=None, a_max=np.quantile(x, quantile_cutoff))

    return tr.stack_plots(
        tr.heatmap_plot(
            _get_heatmap,
            palette=palette,
            zorder=1,
            **heatmap_kw,
        ),
        topography(
            transformer, 
            palette=palette,
            zorder=0,
            alpha=0.15,
            **topography_kw,
        ),
        label=label,
        height=height,
    )


def gene_expression_track(
    expression_key = "GeneExpression",
    strand_key = "GeneStrand",
    linewidth=0.5,
    label="Gene\nexpression",
    color="lightgrey",
    height=1,
    log1p=True,
):
    import numpy as np

    return tr.bar_plot(
        tr.pipeline(
            tr.feature_matrix(
                expression_key,
                strand_key
            ),
            lambda x : np.prod(x, axis=0),
            lambda x : np.sign(x) * np.log1p(np.abs(x)) if log1p else x, #symlog1p
        ),
        ax_fn=lambda ax : (
            ax.axhline(0, color="k", linewidth=linewidth,),
            ax.spines["bottom"].set_visible(False),
        ),
        label=label,
        color=color,
        height=height
    )


def order_components(dataset):
    component_order = (
        tr.pipeline(
            tr.select("component_distributions_locus"),
            tr.apply_rows(tr.renorm),
            lambda x: x.to_pandas(),
            tr.reorder_df,
        )(dataset)
        .index.values
    )
    return component_order


def component_rate_summary(
    view,
    *,
    ideogram,
    scalebar_size=int(1e7),
    scalebar_scale="mb",
    pred_smooth=20,
    empirical_smooth=10,
    legend=True,
    pred_kw={"color": categorical_palette[1], "dashes": (1, 1), "alpha": 1, "linewidth": 0.75},
    component_smooth=30,
    component_order=None
):
    component_order = (
        order_components(view.dataset)
        if component_order is None
        else component_order
    )

    return (
        tr.scale_bar(scalebar_size, scale=scalebar_scale),
        tr.ideogram(ideogram, height=0.1),
        tr.tracks.marginal_observed_vs_expected(
            view, 
            smooth=empirical_smooth,
            pred_smooth=pred_smooth,
            predicted_kw=pred_kw,
            legend=legend
        ),
        tr.spacer(0.1),
        *tr.tracks.component_rates(view, *component_order, smooth=component_smooth)
    )
