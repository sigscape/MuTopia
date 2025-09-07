from math import sqrt
from itertools import chain
from typing import Union

def _plot_linear_signature(
    xlabels,
    palette,
    ax=None,
    legend_title=None,
    height=1.25,
    width=5.25,
    plot_kw={},
    label_xaxis=True,
    **signatures,
):
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    plot_kw = dict(
        width=1,
        edgecolor="white",
        linewidth=0.4,
        error_kw={"alpha": 0.5, "linewidth": 0.5},
        **plot_kw,
    )

    extent = max(chain(*signatures.values()))
    n_sigs = len(signatures)
    sig_dim = len(next(iter(signatures.values())))
    n_bars = sig_dim * n_sigs

    def _get_color(i):
        if isinstance(palette, str):
            return plt.get_cmap(palette)(i)
        elif len(palette) < sig_dim:
            return palette[i % len(palette)]
        else:
            return palette

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(width * n_sigs, height * sqrt(n_sigs)))

    for i, (label, s) in enumerate(signatures.items()):
        ax.bar(
            height=[v / extent for v in s],
            x=range(i, n_bars, n_sigs),
            color=_get_color(i),
            **plot_kw,
            label=label,
        )

    ax.axhline(0, color="black", linewidth=0.5)
    for s in [
        "left",
        "right",
        "top",
    ]:
        ax.spines[s].set_visible(False)

    ax.set(
        xlim=(-0.5, n_bars - 0.5),
        ylim=(0, 1.1),
        yticks=[],
    )

    n_sections = len(xlabels) + 1
    ax.xaxis.set_major_locator(ticker.LinearLocator(n_sections))
    ax.xaxis.set_minor_locator(ticker.LinearLocator(2 * n_sections - 1))
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_formatter(
        ticker.FuncFormatter(lambda x, pos: xlabels[pos] if label_xaxis else "")
    )
    ax.tick_params(axis="x", which="minor", tick1On=False, tick2On=False, labelsize=7)

    if len(signatures) > 1:
        ax.legend(
            title=legend_title.replace("_", " ").title() if legend_title else None,
            fontsize=8,
            ncol=1,
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
        )

    return ax


def plot_spectrum(signature, *select, **kw):
    """
    Plot a component of a signature using its modality's plotting method.

    This function serves as a convenience wrapper that calls the appropriate plotting
    method based on the signature's modality. If no selection is provided, it defaults
    to plotting the "Baseline" component.

    Parameters
    ----------
    signature : xr.DataArray
        The signature data array to plot. Must have a modality method that returns
        an object with a plot method.
    *select : str, optional
        Variable number of string arguments specifying which components to select
        for plotting. If no arguments are provided, defaults to ["Baseline"].
    **kw : dict, optional
        Additional keyword arguments passed to the underlying plot method.

    Returns
    -------
    object
        The return value from the modality's plot method, typically a matplotlib
        figure or axes object.

    Examples
    --------
    >>> plot_component(signature_data)  # Plots baseline component
    >>> plot_component(signature_data, "Component1", "Component2")  # Plots specific components
    >>> plot_component(signature_data, "Baseline", figsize=(10, 6))  # With additional kwargs
    """
    return signature.modality().plot(signature, *select, **kw)


def plot_component(
    dataset,
    component: Union[str, int],
    *select,
    **kw,
):
    """
    Plot a specific component from the dataset using its modality's plotting method.

    This function retrieves the specified component from the dataset and calls its
    modality's plot method. If no selection is provided, it defaults to plotting
    the "Baseline" component.

    Parameters
    ----------
    dataset : xr.DataSet
        The dataset containing the signature components.
    component : int or str
        The index or identifier of the component to plot.
    *select : str, optional
        Variable number of string arguments specifying which components to select
        for plotting. If no arguments are provided, defaults to ["Baseline"].
    key : str, optional
        The key in the dataset where the signatures are stored. Defaults to "mutopia".
    **kw : dict, optional
        Additional keyword arguments passed to the underlying plot method.

    Returns
    -------
    object
        The return value from the modality's plot method, typically a matplotlib
        figure or axes object.

    Examples
    --------
    >>> plot_component(dataset, 0)  # Plots baseline of component at index 0
    >>> plot_component(dataset, "Component1", "Component2")  # Plots specific components by name
    >>> plot_component(dataset, 1, figsize=(10, 6))  # With additional kwargs
    """
    from mutopia.gtensor import fetch_component

    signature = fetch_component(dataset, component)

    if len(select) == 0:
        select = ["Baseline"]
    return signature.modality().plot(signature, *select, **kw)


def plot_signature_panel(
    dataset,
    ncols=4,
    width=3.5,
    height=1.25,
    show=True,
    **kwargs,
):
    """
    Create a panel of signature plots for all components in the model.

    Parameters
    ----------
    ncols : int, default=3
        Number of columns in the panel.
    normalization : str, default="global"
        Normalization method for the signatures.
    width : float, default=3.5
        Width of each subplot in inches.
    height : float, default=1.25
        Height of each subplot in inches.
    show : bool, default=True
        If True, displays the figure. If False, returns the figure object.
    **kwargs
        Additional keyword arguments passed to plot_component method.

    Returns
    -------
    fig : matplotlib.figure.Figure, optional
        The figure object containing the panel of signatures. Only returned if show=False.

    Notes
    -----
    This method creates a grid of subplots, each displaying one signature component.
    The number of rows is calculated based on ncols and the number of components.
    Component names are displayed as y-axis labels.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mutopia.gtensor import fetch_component

    K = len(dataset.coords["component"])
    component_names = dataset.coords["component"].values
    nrows = int(np.ceil(K / ncols))

    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(width * ncols, height * nrows),
        gridspec_kw={"hspace": 0.5, "wspace": 0.25},
    )

    for k in range(K):
        _ax = np.ravel(ax)[k]
        plot_spectrum(fetch_component(dataset, k), ax=_ax, **kwargs)
        _ax.set_ylabel(component_names[k], fontsize=8)

    for _ax in np.ravel(ax)[K:]:
        _ax.axis("off")

    if show:
        plt.show()
    else:
        return fig
