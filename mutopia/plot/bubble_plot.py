import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ..utils import diverging_palette
from ..gtensor import get_shap_summary

def _l2_normalize(x):
    """L2 normalize the input array."""
    norm = np.linalg.norm(x, keepdims=True)
    return x / norm if norm.all() else x

def plot_shap_summary(
    data,
    cmap=diverging_palette,
    figsize=(8, 5),
    scale=100,
    feature_order=None,
    component_order=None,
    ax=None,
    cbar=True,
    max_size=1000,
    linewidth=0.5,
    **scatter_kw,
):

    from scipy.cluster.hierarchy import linkage, leaves_list

    """
    Generate a bubble heatmap for SHAP effect size and correlation.
    """
    component_summary = get_shap_summary(data)

    # Pivot the data for the heatmap
    effect_size_pivot = component_summary.pivot(
        index="component", columns="feature", values="effect_size"
    )
    effect_size_pivot = effect_size_pivot.fillna(0) ** (3 / 2)

    correlation_pivot = component_summary.pivot(
        index="component", columns="feature", values="correlation"
    )
    correlation_pivot = correlation_pivot**2 * np.sign(
        correlation_pivot
    )  # Square to ensure positive correlation values

    # 1. Cluster components (rows)
    # Combine effect size and correlation for component clustering
    component_data = np.hstack(
        [
            _l2_normalize(effect_size_pivot.values),
            _l2_normalize(correlation_pivot.values),
        ]
    )
    # Perform clustering and get the new order
    if component_order is None:
        component_linkage = linkage(component_data, method="ward", metric="euclidean")
        component_order = leaves_list(component_linkage)
    else:
        component_order = effect_size_pivot.index.get_indexer(component_order)

    # 2. Cluster features (columns)
    # Combine effect size and correlation for feature clustering (note the transpose)
    feature_data = np.hstack(
        [
            _l2_normalize(effect_size_pivot.T.values),
            _l2_normalize(correlation_pivot.T.values),
        ]
    )
    # Perform clustering and get the new order

    if feature_order is None:
        feature_linkage = linkage(feature_data, method="ward", metric="euclidean")
        feature_order = leaves_list(feature_linkage)
    else:
        feature_order = effect_size_pivot.columns.get_indexer(feature_order)

    # 3. Reorder the pivot tables and labels
    effect_size_pivot = effect_size_pivot.iloc[component_order, feature_order]
    correlation_pivot = correlation_pivot.iloc[component_order, feature_order]
    features = effect_size_pivot.columns
    components = effect_size_pivot.index

    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Use a scatterplot to draw the bubbles
    # `x` and `y` are the grid coordinates
    # `hue` is for color (correlation)
    # `size` is for bubble size (effect_size)
    x, y = np.meshgrid(np.arange(features.size), np.arange(components.size))
    
    scatter = ax.scatter(
        x=x.flatten(),
        y=y.flatten(),
        s=np.minimum(effect_size_pivot.values.flatten(), max_size) * scale,
        c=correlation_pivot.values.flatten(),
        cmap=cmap,
        vmin=-1,
        vmax=1,
        linewidth=linewidth,
        edgecolor="black",
        **scatter_kw,
    )

    ax.grid(color="lightgrey", linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    # Add plot labels and ticks
    ax.set_xticks(np.arange(features.size))
    ax.set_xticklabels(features, rotation=45, ha="left")
    ax.set_yticks(np.arange(components.size))
    ax.set_yticklabels(components)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Component")

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", which="both", length=0)
    ax.tick_params(axis="y", which="both", length=0)

    if cbar:
        # Create legends for both color and size
        cbar = fig.colorbar(scatter, ax=ax, shrink=max(0.5, 2 / figsize[1]))
        cbar.set_label("Feature-SHAP\nCorrelation", fontsize=9)
        cbar.set_ticks([-1, 0, 1])
        # Reduce the thickness of the colorbar ticks
        cbar.ax.tick_params(width=0.5)
        cbar.outline.set_linewidth(0.5)

    ax.set(ylim=(-1, components.size), xlim=(-0.5, features.size))
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    return ax
