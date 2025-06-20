from math import sqrt
from itertools import chain
import matplotlib.pyplot as plt
from matplotlib import ticker
from ..utils import borrow_kwargs


def _plot_linear_signature(
    xlabels,
    palette,
    ax=None,
    legend_title=None,
    height=1.25,
    width=5.25,
    plot_kw={},
    **signatures,
):

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
    ax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, pos: xlabels[pos]))
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


@borrow_kwargs(_plot_linear_signature)
def plot_component(signature, **kw):
    return signature.modality().plot(signature, **kw)
