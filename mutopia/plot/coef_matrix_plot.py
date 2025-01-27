import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec

def _plot_interaction_matrix(
        interaction_matrix : pd.DataFrame,
        baseline_matrix : pd.Series,
        shared_effects : pd.Series,
        palette='coolwarm',
        gridspec=None,
    ):

    n_rows, _ = interaction_matrix.shape
    plot_height = 1 + 0.35 * n_rows

    gs_kw = dict(
        width_ratios=[0.22,7, 0.1], 
        height_ratios=[1, plot_height-1],
        wspace=0.05,
        hspace=0.1,
    )

    if gridspec is None:
        fig = plt.figure(figsize=(10, plot_height))
        gs = plt.GridSpec(2,3, **gs_kw)
    else:
        fig = plt.gcf()
        gs = gridspec.subgridspec(2,3, **gs_kw)

    base_ax = fig.add_subplot(gs[0,1])
    baseline = baseline_matrix
    baseline_x = np.arange(len(baseline))

    base_ax.bar(
        baseline_x,
        height=baseline.values.ravel(),
        color='lightgray',
        edgecolor='grey',
        linewidth=0.5
    )
    for spine in ['left','right','top']:
        base_ax.spines[spine].set_visible(False)
    base_ax.set(
        yticks=[],
        xlim = (-0.5, len(baseline)-0.5),
        xticks=[],
    )
    base_ax.set_ylabel('Base\nmutation rate', rotation=0, labelpad=0.1, fontsize=8, ha='right', va='center')


    interaction_ax = fig.add_subplot(gs[1,1])
    interactions = interaction_matrix

    extrema = max(np.max(interaction_matrix.abs()), np.max(shared_effects.abs()), 0.5)

    heat_x = np.arange(interactions.shape[0]) - 0.5
    heat_y = np.arange(interactions.shape[1]) - 0.5
    interaction_ax.pcolormesh(
        heat_y,
        heat_x,
        interactions.values,
        cmap=palette,
        shading='auto',
        rasterized=True,
        vmin=-extrema,
        vmax=extrema,
        edgecolor='white',
        linewidth=0.1,
    )
    interaction_ax.set_xticks(baseline_x - 0.5)
    interaction_ax.set_xticklabels(baseline.index, rotation=90, fontsize=5)
    interaction_ax.set(yticks=[])
    interaction_ax.set_xlabel('Nucleotide context', fontsize=8)
    for spine in interaction_ax.spines.values():
        spine.set_visible(False)

    common_ax = fig.add_subplot(gs[1,0])
    common_x = np.arange(2)
    common_y = np.arange(interaction_matrix.shape[0] + 1) - 0.5

    common_ax.pcolormesh(
        common_x,
        common_y,
        shared_effects.values[:,None],
        cmap=palette,
        vmin=-extrema,
        vmax=extrema,
        edgecolor='white',
        linewidth=0.1,
    )
    for spine in common_ax.spines.values():
        spine.set_visible(False)
    common_ax.set_yticks(np.arange(interaction_matrix.shape[0]))
    common_ax.set_yticklabels(interaction_matrix.index, fontsize=8)
    common_ax.set(xticks=[0.5])
    common_ax.set_ylabel('Features', fontsize=8)
    common_ax.set_xticklabels(['Shared\neffect'], rotation=90, fontsize=8)

    cbar_ax = fig.add_subplot(gs[1, 2])
    cbar = fig.colorbar(interaction_ax.collections[0], 
                        cax=cbar_ax, orientation='vertical',
                        )
    cbar.set_label('Interaction effect', rotation=90, labelpad=5, fontsize=8)
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=8)

    return gs