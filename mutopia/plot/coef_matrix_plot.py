import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial
from typing import Union
from ..utils import diverging_palette
from ..gtensor import fetch_interactions, fetch_component, fetch_shared_effects


def _plot_interaction_matrix(
    signature_plot_fn,
    interaction_matrix: pd.DataFrame,
    shared_effects: pd.Series,
    palette=diverging_palette,
    gridspec=None,
):

    n_rows, _ = interaction_matrix.shape
    plot_height = 1 + 0.35 * n_rows

    gs_kw = dict(
        width_ratios=[0.22, 7, 0.1],
        height_ratios=[1, plot_height - 1],
        wspace=0.05,
        hspace=0.25,
    )

    if gridspec is None:
        fig = plt.figure(figsize=(10, plot_height))
        gs = plt.GridSpec(2, 3, **gs_kw)
    else:
        fig = plt.gcf()
        gs = gridspec.subgridspec(2, 3, **gs_kw)

    base_ax = fig.add_subplot(gs[0, 1])
    signature_plot_fn(ax=base_ax)
    base_ax.set_ylabel(
        "Base\nmutation ", rotation=0, labelpad=0.1, fontsize=8, ha="right", va="center"
    )
    base_ax.set_xticklabels([])

    interaction_ax = fig.add_subplot(gs[1, 1])
    interactions = interaction_matrix

    extrema = max(np.max(interaction_matrix.abs()), np.max(shared_effects.abs()), 0.5)

    heat_x = np.arange(interactions.shape[0]) - 0.5
    heat_y = np.arange(interactions.shape[1]) - 0.5
    interaction_ax.pcolormesh(
        heat_y,
        heat_x,
        interactions.values,
        cmap=palette,
        shading="auto",
        rasterized=True,
        vmin=-extrema,
        vmax=extrema,
        edgecolor="white",
        linewidth=0.1,
    )

    interaction_ax.set(yticks=[], xticks=[])
    interaction_ax.set_xlabel("Context", fontsize=8)
    for spine in interaction_ax.spines.values():
        spine.set(edgecolor="lightgrey", linewidth=0.5)

    common_ax = fig.add_subplot(gs[1, 0])
    common_x = np.arange(2)
    common_y = np.arange(interaction_matrix.shape[0] + 1) - 0.5

    common_ax.pcolormesh(
        common_x,
        common_y,
        shared_effects.values[:, None],
        cmap=palette,
        vmin=-extrema,
        vmax=extrema,
        edgecolor="white",
        linewidth=0.1,
    )
    for spine in common_ax.spines.values():
        spine.set(edgecolor="lightgrey", linewidth=0.5)

    common_ax.set_yticks(np.arange(interaction_matrix.shape[0]))
    common_ax.set_yticklabels(interaction_matrix.index, fontsize=8)
    common_ax.set(xticks=[0.5])
    common_ax.set_ylabel("Features", fontsize=8)
    common_ax.set_xticklabels(["Shared\neffect"], rotation=90, fontsize=8)

    cbar_ax = fig.add_subplot(gs[1, 2])
    cbar = fig.colorbar(
        interaction_ax.collections[0],
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.set_label("Interaction effect", rotation=90, labelpad=5, fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    return gs



def plot_interaction_matrix(
    dataset,
    component : Union[str, int],
    palette=diverging_palette,
    gridspec=None,
    **kw,
):
    """
    Generate a visualization of component interactions.

    This method creates a plot showing the interaction matrix for a specified component.
    It displays shared effects and context-specific interactions for genomic signatures.

    Parameters
    ----------
    dataset : xr.DataSet
        The dataset containing the interactions to visualize.
    component : int or str
        The component index or identifier to visualize.
    palette : function, optional
        A color palette function to use for visualization, defaults to diverging_palette.
    normalization : str, optional
        Method for normalizing the signature values.
    **kw : dict
        Additional keyword arguments passed to the underlying plotting function.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the visualization of the interaction matrix.

    Notes
    -----
    The interaction matrix shows how the component behaves across different contexts,
    highlighting both shared effects and context-specific variations.
    """
    interactions = (
        fetch_interactions(dataset, component)
        .drop_sel(genome_state="Baseline")
    )
    dtype = interactions.modality()
    interactions = dtype._flatten_observations(interactions).to_pandas()

    shared_effects = (
        fetch_shared_effects(dataset, component)
        .drop_sel(genome_state="Baseline")
        .to_pandas()
    )
    
    signature = fetch_component(dataset, component)

    return _plot_interaction_matrix(
        partial(dtype.plot, signature, "Baseline"),
        interactions,
        shared_effects,  # .iloc[:,0],
        palette=palette,
        gridspec=gridspec,
        **kw,
    )