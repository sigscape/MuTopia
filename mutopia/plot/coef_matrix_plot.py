from __future__ import annotations

from functools import partial
from typing import Any, Optional, Union, TYPE_CHECKING
from mutopia.palettes import diverging_palette

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec
    from mutopia.gtensor.gtensor import GTensorDataset


def _plot_interaction_matrix(
    signature_plot_fn,
    interaction_matrix,
    shared_effects,
    palette=diverging_palette,
    gridspec: Optional["GridSpec"] = None,
    title: Optional[str] = None,
    base_height: float = 1.5,
    heatmap_row_height: float = 0.25,
    width: float = 10,
):
    import numpy as np
    import matplotlib.pyplot as plt

    n_rows, _ = interaction_matrix.shape
    plot_height = base_height + heatmap_row_height * n_rows

    gs_kw = dict(
        width_ratios=[0.22, 7, 0.1],
        height_ratios=[base_height, plot_height - base_height],
        wspace=0.05,
        hspace=0.25,
    )

    if gridspec is None:
        fig = plt.figure(figsize=(width, plot_height))
        gs = plt.GridSpec(2, 3, **gs_kw)
    else:
        fig = plt.gcf()
        gs = gridspec.subgridspec(2, 3, **gs_kw)

    base_ax = fig.add_subplot(gs[0, 1])
    signature_plot_fn(ax=base_ax)
    base_ax.set_ylabel(
        title or "Base \nrates ",
        rotation=0,
        labelpad=0.1,
        fontsize=8,
        ha="right",
        va="center",
    )
    base_ax.set_xticklabels([])

    interaction_ax = fig.add_subplot(gs[1, 1])
    interactions = interaction_matrix

    extrema = max(np.max(interaction_matrix.abs()), np.max(shared_effects.abs()), 0.5)

    heat_x = np.arange(interactions.shape[0]) - 0.5
    heat_y = np.arange(interactions.shape[1]) - 0.5
    if interactions.shape[0] == 1:
        interaction_ax.pcolormesh(
            interactions.values,
            cmap=palette,
            shading="auto",
            rasterized=True,
            vmin=-extrema,
            vmax=extrema,
            edgecolor="white",
            linewidth=0.1,
        )

    else:
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
    dataset: "GTensorDataset",
    component: Union[str, int],
    palette=diverging_palette,
    gridspec: Optional["GridSpec"] = None,
    title: Optional[str] = None,
    **kw: Any,
) -> "GridSpec":
    """
    Generate a visualization of component interactions.

    This method creates a plot showing the interaction matrix for a specified component.
    It displays shared effects and context-specific interactions for genomic signatures.

    Parameters
    ----------
    dataset : GTensorDataset
        Dataset containing the interactions to visualize.
    component : int or str
        The component index or identifier to visualize.
    palette : function, optional
        A color palette function to use for visualization, defaults to diverging_palette.
    gridspec : matplotlib.gridspec.GridSpec, optional
        GridSpec to draw into; when None, a new Figure is created and used.
    title : str, optional
        Label for the base-rate row.
    **kw : dict
        Extra keyword arguments forwarded to the modality plot function.

    Returns
    -------
    matplotlib.gridspec.GridSpec
        The sub-GridSpec used for the interaction plot layout.

    Notes
    -----
    The interaction matrix shows how the component behaves across different contexts,
    highlighting both shared effects and context-specific variations.
    """
    from mutopia.gtensor import (
        fetch_interactions,
        fetch_component,
        fetch_shared_effects,
    )

    interactions = fetch_interactions(dataset, component).drop_sel(
        genome_state="Baseline"
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
        partial(dtype.plot, signature, "Baseline", label_xaxis=False),
        interactions,
        shared_effects,  # .iloc[:,0],
        palette=palette,
        gridspec=gridspec,
        title=title,
        **kw,
    )
