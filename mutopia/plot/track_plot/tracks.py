from .track_plot import static_track
import mutopia.plot.track_plot as tr
from mutopia.utils import categorical_palette
from functools import partial

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
    predicted_kw={"color": "black", "dashes": (1, 1), "alpha": 0.5, "linewidth": 0.75},
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


def _get_topography(
    dataset,
    vmin=-3,
    vmax=3,
    mutation_order=['C>G', 'C>A','T>A','T>C','T>G','C>T'],
):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    topograph = (
        dataset['predicted_marginal']/dataset['predicted_marginal_locus']
    ).stack(observation=('context','configuration')).transpose('locus',...).to_pandas().T
    
    topograph = pd.DataFrame(
        np.clip(
            StandardScaler().fit_transform( 
                np.log(topograph).values.reshape(96, -1).T
            ), 
            vmin, vmax,
        ).T.reshape(192, -1),
        index=topograph.index,
        columns=topograph.columns
    )
    
    mutation = topograph.index.get_level_values('context').str.slice(2,5)
    topograph['mutation'] = mutation
    topograph = topograph.reset_index().set_index(['configuration','mutation','context'])

    context_order=[]
    for mutation_type in mutation_order:
        df=topograph.loc['C/T-centered', mutation_type]
        context_order.extend(tr.reorder_df(df).index.values)
    
    topography_order = [
        (center, context)
        for center in ['C/T-centered', 'A/G-centered']
        for context in (context_order[::-1] if center == 'C/T-centered' else context_order)
    ]
    
    return topograph.droplevel(1).loc[topography_order].values


def topography(
    vmin=-3,
    vmax=3,
    palette="Greys",
    yticks=False,
    label='Predicted\ntopography',
    **kw,
):
    return tr.heatmap_plot(
        partial(_get_topography, vmin=vmin, vmax=vmax),
        palette=palette,
        yticks=yticks,
        label=label,
        **kw,
    )


def gene_expression_track(
    expression_key = "GeneExpression",
    strand_key = "GeneStrand",
    linewidth=0.5,
    label="Gene\nexpression",
    color="lightgrey",
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
    )