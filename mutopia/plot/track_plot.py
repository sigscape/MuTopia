
import numpy as np
import matplotlib.pyplot as plt
import os
from functools import partial
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from ..genome_utils.bed12_utils import unstack_regions
from ..utils import borrow_kwargs

def moving_average(arr, bin_width=None, alpha=10):
    if bin_width is None:
        weights = np.ones(alpha)/alpha
        ema = np.convolve(arr, weights, mode='same')
    else:
        # Fix moving average rate to weighted average rate to use sum(bin width * rate)/ (total bin width) 
        window = np.ones(alpha)
        weighted_sum = np.convolve(arr * bin_width, window, mode='same')
        total_weight = np.convolve(bin_width, window, mode='same')

        # Compute the weighted moving average
        ema = weighted_sum / total_weight  

    return ema

def renorm(x):
    return x/x.sum()

def reorder_rows(data, labels):
    row_idx = leaves_list(optimal_leaf_ordering(linkage(data, method='ward'), data))
    return {
        'matrix' : data[row_idx],
        'row_labels' : labels[row_idx],
    }

def _trace_plot(
    start, end, idx, check_len,
    *plotting_fns,
    width=10,
):
    _, ax = plt.subplots(
        len(plotting_fns), 1, 
        figsize=(width, sum(track.height for track in plotting_fns)),
        height_ratios=[track.height for track in plotting_fns],
        gridspec_kw={'hspace': 0.1},
    )

    xlim = (min(start), max(end))

    if len(plotting_fns) == 1:
        ax = [ax]
    
    for _ax, fn in zip(ax, plotting_fns):
        fn(_ax, start, end, idx, check_len).set(xlim=xlim)

    return ax


def make_view(corpus):

    n_regions = corpus.coords['locus'].size

    start, end, idx = unstack_regions(
        corpus.coords['locus'].values,
        os.path.join(os.path.dirname(corpus.attrs['filename']), corpus.attrs['regions_file']),
        np.arange(n_regions),
    )

    return partial(_trace_plot, start, end, idx, n_regions)


def _set_axlabel(ax, label):
    ax.set_ylabel(label, rotation=0, labelpad=5, fontsize=9, ha='right', va='center')


def _clean_ax(ax, axlabel=None):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set(xticks=[])
    if not axlabel is None:
        _set_axlabel(ax, axlabel)


def stack_plots(*plotting_fns, height=1, label=None):
    def _stack(ax, *args, **kwargs):
        for fn in plotting_fns:
            fn(ax, *args, **kwargs)
        if not label is None:
            _set_axlabel(ax, label)
        return ax
    _stack.height = height
    return _stack


def xaxis_plot(height=0.1):
    def _plot(ax, start, end, idx, check_len):
        ax.tick_params(
            axis='x', 
            top=True, labeltop=True, 
            bottom=False, labelbottom=False,
            labelsize=7
        )
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set(yticks=[])
        ax.set_title('Genomic Position', fontsize=9)
        return ax

    _plot.height = height
    return _plot


@borrow_kwargs(plt.plot)
def line_plot(
    vals,
    label=None,
    height=1,
    **kwargs,
):
    def _plot(ax, start, end, idx, check_len):

        if not len(vals) == check_len:
            raise ValueError('vals and must have one entry per region in the provided corpus.')
        
        x = np.empty((start.size + end.size,), dtype=start.dtype)
        x[0::2] = start
        x[1::2] = end
        ax.plot(x, np.repeat(vals[idx], 2), **kwargs)
        _clean_ax(ax, axlabel=label)    
        return ax
    
    _plot.height = height
    return _plot


@borrow_kwargs(plt.bar)
def bar_plot(
    vals,
    label=None,
    height=1,
    **kwargs,
):
    def _plot(ax, start, end, idx, check_len):
        
        if not len(vals) == check_len:
            raise ValueError('vals and must have one entry per region in the provided corpus.')
        
        center = start + (end - start)/2
        width = end - start
        ax.bar(center, vals[idx], width=width, **kwargs)
        _clean_ax(ax, axlabel=label)
        return ax
    
    _plot.height = height
    return _plot


@borrow_kwargs(plt.scatter)
def scatter_plot(
    vals,
    label=None,
    height=1,
    **kwargs,
):
    def _plot(ax, start, end, idx, check_len):
        
        if not len(vals) == check_len:
            raise ValueError('vals and must have one entry per region in the provided corpus.')
        
        ax.scatter(start + (end - start)/2, vals[idx], **kwargs)
        _clean_ax(ax, axlabel=label)
        return ax
    
    _plot.height = height
    return _plot


@borrow_kwargs(plt.pcolormesh)
def heatmap_plot(
    matrix,
    row_labels = [],
    palette='crest_r',
    label=None,
    height=1,
):
    def _plot(ax, start, end, idx, check_len):
        
        if not matrix.shape[1] == check_len:
            raise ValueError('The passed matrix must have the same number of columns as there are regions in the corpus. Perhaps try transposing!.')
        
        nrows = matrix.shape[0]
        x=np.arange(nrows+1)
        y=np.concatenate([[start[0]], end])
        z = matrix[:, idx]

        ax.pcolormesh(y, x, z, cmap=palette)
        ax.set_yticks(np.arange(nrows)+0.5)
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set(xticks=[])
        if not label is None:
            _set_axlabel(ax, label)
        return ax
    
    _plot.height = height
    return _plot


def categorical_plot(
    vals, 
    order=None,
    label=None, 
    height=1,
    palette='Greys', 
):
    
    categories = order or list(sorted(set(vals)))

    if isinstance(palette, str):
        palette = plt.get_cmap(palette)(np.linspace(0.1, 0.9, len(categories)))
    elif not len(palette) == len(categories):
        raise ValueError('Palette must have the same number of colors as there are categories.')
    
    def _plot(ax, start, end, idx, check_len):
        
        if not len(vals) == check_len:
            raise ValueError('vals and must have one entry per region in the provided corpus.')

        for _start, _end, _val in zip(
            start, end, vals[idx]
        ):
            if _val in categories:
                try:
                    j = categories.index(_val)
                except ValueError:
                    continue
                
                ax.add_patch(plt.Rectangle(
                    (_start, 0.1*j), _end - _start, 0.1,
                    color=palette[j],
                    edgecolor=None,
                ))

        ax.set(
            yticks=list(0.1*np.arange(len(categories)) + 0.05),
            yticklabels=[str(c) for c in categories],
            xticks=[],
            ylim=(0, 0.1*len(categories)),
        )
        for spine in ax.spines.values():
            spine.set_visible(False)

        if not label is None:
            _set_axlabel(ax, label)
        return ax
    
    _plot.height = height
    return _plot
