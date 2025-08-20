import numpy as np
from functools import partial
import mutopia.plot.track_plot as tr
from mutopia.utils import categorical_palette, diverging_palette
from .track_plot import static_track
from .transforms import TopographyTransformer, minmax_scale

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
    smooth=20,
    pred_smooth=10,
    label="Mutation rate",
    legend=True,
    height=1,
    empirical_kw={"alpha": 0.5, "s": 0.1, "color": "lightgrey"},
    predicted_kw={"color": categorical_palette[1], "dashes": (1, 1), "alpha": 0.8, "linewidth": 0.75},
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


def mega_summary(
    view,*, 
    dataset, 
    mutation_rate_plot_kw={},
    shap_plot_kw={},
): 

    import seaborn as sns
    from mutopia.plot import plot_component, plot_shap_summary
    from mutopia.gtensor import fetch_component

    feature_order=[
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
            dataset
            .sections["Spectra"]["shared_effects"]
            .sel(component=component_name)
            .to_pandas().iloc[1:,]
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
            spine.set_color('lightgrey')
            spine.set_linewidth(0.5)
            
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)


    def _plot_component(comp_name, ax):
        ax = plot_component(fetch_component(dataset, comp_name), ax=ax)
        ax.set_ylabel(comp_name, fontsize=9, rotation=0, labelpad=10, va="center", ha="right")
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

    phases = ["G1b", "S1", "S2", "S3", "S4","G2"]

    feature_hm = tr.heatmap_plot(
        tr.pipeline(
            tr.feature_matrix(*feature_order),
            tr.clip(0, 0.97),
            lambda x : np.log1p(x),
            tr.apply_rows(minmax_scale),
            lambda x : x.fillna(0.),
        ),
        palette="viridis",
        cbar=False,
        label="Functional\nfeatures"
    )
    
    repliseq = tr.heatmap_plot(
        tr.pipeline(
            tr.feature_matrix(*["Repliseq" + phase for phase in phases]),
            tr.apply_rows(minmax_scale),
        ),
        palette="crest_r",
        cbar=False,
        label="Cell cycle\nphase",
        ax_fn= lambda ax : (
            ax.set_yticklabels(reversed(phases))
        )
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
            tr.ideogram('/Users/allen/projects/mutopia/signaturemodels/notebooks/cytoBand.txt', height=0.1),
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
    
    view = tr.make_view(dataset, 'chr1:49007569-69616441', title=None)

    return tr.plot_view(
        mega_summary,
        view,
        dataset=dataset,
        **kw,
        width_ratios=[3,4,1,4], 
        width=11, 
        gridpsec_kw = {"hspace": 0.3, "wspace": 0.2}
    )
