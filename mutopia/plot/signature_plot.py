
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt


def _plot_linear_signature(
    xlabels,
    palette,
    signature, 
    *compare_sigs,
    sig_names=None,
    ax=None, 
    height = 1,
    width = 4.25,
    **kwargs,
):
    
    sig_dim = len(signature)
    
    plot_kw = dict(
        width = 1,
        edgecolor = 'white',
        linewidth = 0.5,
        error_kw = {'alpha' : 0.5, 'linewidth' : 0.5},
        **kwargs,
    )

    extent=max(chain(signature, *compare_sigs))
    n_sigs = len(compare_sigs) + 1

    if ax is None:
        _, ax = plt.subplots(1,1,figsize=(width*n_sigs, height*n_sigs))

    if n_sigs==1:
        signature=np.array(signature)
        signature=signature/extent
        ax.bar(
            height = signature, 
            x = range(len(xlabels)),
            color = palette,
            **plot_kw
        )
    else:
        for i, s in enumerate((signature, *compare_sigs)):
            ax.bar(
                height = np.array(s)/extent, 
                x = range(i, n_sigs*len(xlabels), n_sigs),
                color = plt.get_cmap(palette)(i) if isinstance(palette, str) else palette[i],
                **plot_kw,
                label = sig_names[i] if not sig_names is None else f'Signature {i+1}'
            )

    ax.set(
            yticks = [0.25, 0.5, 0.75, 1.], 
            xticks = [],
            xlim = (-1, sig_dim*n_sigs + 1),
            ylim = (0, 1.1)
        )

    ax.axhline(0, color = 'black', linewidth = 0.5)

    for s in ['left','right','top','bottom']:
        ax.spines[s].set_visible(False)

    ## add a whitegrid at the quarter, half, and top marks of the y-axis
    ax.yaxis.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_yticklabels([], fontsize=8)

    # keep the grid but remove the ytick markers
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    if len(compare_sigs) > 0:
        ax.legend(loc='lower center', fontsize=8, ncol=n_sigs,
                    bbox_to_anchor=(0.5, -0.25), frameon=False)
        
    return ax
