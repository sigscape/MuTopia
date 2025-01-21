
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt


def _plot_linear_signature(
    xlabels,
    palette,
    ax=None,
    legend_title=None,
    height = 1.25,
    width = 5.25,
    plot_kw = {},
    **signatures,
):    
    plot_kw = dict(
        width = 1,
        edgecolor = 'white',
        linewidth = 0.5,
        error_kw = {'alpha' : 0.5, 'linewidth' : 0.5},
        **plot_kw,
    )

    extent=max(chain(*signatures.values()))
    n_sigs = len(signatures)
    sig_dim = len(next(iter(signatures.values())))

    if ax is None:
        _, ax = plt.subplots(1,1,figsize=(width*n_sigs, height*n_sigs))

    for i, (label, s) in enumerate(signatures.items()):
        ax.bar(
            height = np.array(s)/extent, 
            x = range(i, n_sigs*len(xlabels), n_sigs),
            color = plt.get_cmap(palette)(i) if isinstance(palette, str) else palette[i],
            **plot_kw,
            label=label,
        )

    ax.set(
        yticks = [0.25, 0.5, 0.75, 1.], 
        xlim = (-1, sig_dim*n_sigs + 1),
        ylim = (0, 1.1)
    )

    ax.set_xticks(
        ticks = np.arange(0, sig_dim*n_sigs, n_sigs) + 0.5*(n_sigs - 1),
        labels = xlabels,
        fontsize=4,
        fontweight="light",
        rotation=90,
        )

    ax.axhline(0, color = 'black', linewidth = 0.5)

    for s in ['left','right','top',]:
        ax.spines[s].set_visible(False)

    ## add a whitegrid at the quarter, half, and top marks of the y-axis
    ax.yaxis.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_yticklabels([], fontsize=8)

    # keep the grid but remove the ytick markers
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    if len(signatures) > 1:
        ax.legend(
            title=legend_title,
            fontsize=8, 
            ncol=n_sigs,
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
        )
        
    return ax
