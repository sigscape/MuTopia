from __future__ import annotations

from math import sqrt
from itertools import chain
from typing import Any, Optional, Sequence, Union, TYPE_CHECKING

if TYPE_CHECKING:  # Only imported for typing; avoids runtime dependency cycles
    import xarray as xr
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from mutopia.gtensor.gtensor import GTensorDataset

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


def plot_spectrum(
    signature: "xr.DataArray",
    *select: str,
    **kw: Any,
) -> "Axes":
    """
    Plot a component of a signature using its modality's plotting method.

    This wraps the modality-specific plot method. If no selection is provided,
    downstream modality implementations typically default to "Baseline".

    Parameters
    ----------
    signature : xr.DataArray
        Signature data array to plot. Must implement ``signature.modality().plot(...)``.
    *select : str, optional
        One or more state/section labels to plot (e.g., "Baseline").
    **kw : dict, optional
        Extra keyword arguments forwarded to the modality plot method.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the rendered plot.
    """
    return signature.modality().plot(signature, *select, **kw)


def plot_component(
    dataset: "GTensorDataset",
    component: Union[str, int],
    *select: str,
    **kw: Any,
) -> "Axes":
    """
    Plot a specific component from a dataset using its modality's plotting method.

    Parameters
    ----------
    dataset : GTensorDataset
        Dataset containing signature components.
    component : int or str
        Component index or identifier to plot.
    *select : str, optional
        One or more state/section labels to plot (e.g., "Baseline").
    **kw : dict, optional
        Extra keyword arguments forwarded to the modality plot method.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the rendered plot.
    """
    from mutopia.gtensor import fetch_component

    signature = fetch_component(dataset, component)

    if len(select) == 0:
        select = ["Baseline"]
    return signature.modality().plot(signature, *select, **kw)


def plot_signature_panel(
    dataset: "GTensorDataset",
    ncols: int = 4,
    width: float = 3.5,
    height: float = 1.25,
    show: bool = True,
    **kwargs: Any,
) -> Optional["Figure"]:
    """
    Create a panel of signature plots for all components in a dataset.

    Parameters
    ----------
    dataset : GTensorDataset
        Dataset containing signature components.
    ncols : int, default=4
        Number of columns in the panel.
    width : float, default=3.5
        Width of each subplot in inches.
    height : float, default=1.25
        Height of each subplot in inches.
    show : bool, default=True
        If True, display the figure; if False, return it.
    **kwargs
        Extra keyword arguments forwarded to ``plot_spectrum``.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure when ``show=False``; otherwise ``None``.
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
