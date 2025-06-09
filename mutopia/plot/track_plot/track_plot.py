"""
Genome track plotting utilities for visualizing genomic data.

This module provides functions for creating genome browser-style visualizations
with multiple data tracks including line plots, heatmaps, and genomic annotations.
"""
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from functools import partial
from xarray import DataArray, Dataset
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from functools import cache
from dataclasses import dataclass
from typing import Callable, Union, Iterable

from ...genome_utils.bed12_utils import unstack_regions
from ...utils import borrow_kwargs, logger, diverging_palette, parse_region
from ...gtensor.gtensor import slice_regions, get_regions_filename
from .transforms import (
    _moving_average,
    _xarr_op,
)

plt.rc("axes", linewidth=0.75)

def _wraps_err(fn):
    def _inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in plotting function {fn.__name__}: {e}")
            raise e
        
    return _inner


__all__ = [
    "make_view",
    "plot_views",
    "stack_plots",
    "scale_bar",
    "xaxis_plot",
    "spacer",
    "ideogram",
    "line_plot",
    "fill_plot",
    "bar_plot",
    "scatterplot",
    "heatmap_plot",
    "categorical_plot",
    "custom_plot",
]

def _get_optimal_row_order(data, **kwargs):
    return leaves_list(optimal_leaf_ordering(linkage(data, **kwargs), data))

@dataclass
class GenomeView:
    chrom: str
    interval : tuple[int, int]
    dataset: Dataset
    title : str
    n_regions: int
    starts: np.ndarray[int]
    ends: np.ndarray[int]
    idxs: np.ndarray[int]

    def smooth(self, alpha=10):
        smooth_fn = partial(
            _moving_average,
            self.dataset.sections["Regions"].length.values,
            alpha=alpha,
        )

        return _xarr_op(smooth_fn)
    
    def renorm(self, x):
        return x / np.nansum(x) * self.n_regions


def make_view(dataset, region=None, title=None):
    """
    Create a GenomeView for a specific genomic region.

    Parameters
    ----------
    dataset
        Input genomic dataset
    chrom : str
        Chromosome identifier (e.g., 'chr1', 'chrX')
    start : int
        Start genomic coordinate (0-based)
    end : int
        End genomic coordinate (exclusive)

    Returns
    -------
    GenomeView
        Configured genome view object
    """


    if region is None:
        chroms = set(dataset["Regions/chrom"].values)
        if len(chroms) > 1:
            raise ValueError(
                "Chromosome must be specified when dataset contains multiple chromosomes."
            )
        chrom = chroms.pop()
    else:
        chrom = parse_region(region)[0]
        dataset = slice_regions(dataset, region)

    n_regions = dataset.coords["locus"].size

    _, starts, ends, idxs = unstack_regions(
        dataset.coords["locus"].values,
        get_regions_filename(dataset),
        np.arange(n_regions),
    )

    start = min(starts); end = max(ends)

    return GenomeView(
        chrom=chrom,
        interval=(start, end),
        dataset=dataset,
        starts=starts,
        ends=ends,
        idxs=idxs,
        n_regions=n_regions,
        title=str(title) if not title is None else f"{chrom}:{start}-{end}",
    )


'''def __call__(self, configuration, *args, width=7, **kwargs):
    return self._plot(configuration(self, *args, **kwargs), width=width)'''

def plot_views(
    configuration : Callable,
    views : Union[GenomeView, Iterable[GenomeView]],
    width: float = 7,
    gridpsec_kw: dict = {"hspace": 0.1, "wspace": 0.1},
    *args,
    **kwargs,
):

    if not isinstance(views, Iterable):
        views = [views]

    configurations = [configuration(view, *args, **kwargs) for view in views]

    tracks = configurations[0]
    n_rows = len(tracks)
    n_cols = len(views)

    track_names = [fn.track_name for fn in tracks if not fn.track_name is None]

    if not len(track_names) == len(set(track_names)):
        raise ValueError(
            "If supplied, track names must be unique. "
            f"Found duplicates in {track_names}"
        )

    fig, ax = plt.subplots(
        n_rows,
        n_cols,
        figsize=(width, sum(track.height for track in tracks)),
        height_ratios=[track.height for track in tracks],
        gridspec_kw=gridpsec_kw,
        sharex="col",
        sharey="row",
    )

    if n_cols==1:
        ax = np.array(ax).reshape((n_rows, 1))

    if len(tracks) == 1:
        ax = [ax]

    for j, (view, tracks) in enumerate(zip(views, configurations)):
        for i, (_ax, fn) in enumerate(zip(ax[:,j], tracks)):

            fn(
                _ax,
                fig=fig,
                dataset=view.dataset,
                chrom=view.chrom,
                start=view.starts,
                end=view.ends,
                idx=view.idxs,
                check_len=view.n_regions,
            )

            if j > 0:
                _ax.set_ylabel("")

            if j < n_cols - 1:
                try:
                    _ax.get_legend().remove()
                except AttributeError:
                    pass

            _ax.set(xlim=view.interval)

        ax[0,j].set_title(view.title, fontsize=9, loc="left")
    
    return ax


def _set_axlabel(ax, label):
    ax.set_ylabel(label, rotation=0, labelpad=5, fontsize=9, ha="right", va="center")


def _clean_ax(ax, axlabel=None, yticks=False):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if not yticks:
        ax.set(yticks=[])

    ax.set(xticks=[])
    if not axlabel is None:
        _set_axlabel(ax, axlabel)


def stack_plots(
    *plotting_fns, height=1, label=None, legend=True, name=None, ax_fn=lambda x: x
):
    """
    Combine multiple plotting functions into a single track.

    Parameters
    ----------
    *plotting_fns
        Variable number of plotting functions to stack
    height : float, default 1
        Track height in figure units
    label : str, optional
        Track label for y-axis
    legend : bool, default True
        Whether to show legend
    name : str, optional
        Track name for referencing
    ax_fn : callable
        Function to apply final customizations to axes

    Returns
    -------
    callable
        Combined plotting function
    """

    def _plot(ax, **kwargs):

        for fn in plotting_fns:
            fn(ax, **kwargs)

        if not label is None:
            _set_axlabel(ax, label)

        # if any(artist.get_label() for artist in artists):
        if legend:
            ax.legend(
                loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7, frameon=False
            )
        return ax_fn(ax)

    _plot.height = height
    _plot.track_name = name
    return _plot


def scale_bar(length=10000, height=0.1, name=None, label=None):
    """
    Create a scale bar track showing genomic distance.

    Parameters
    ----------
    length : int, default 10000
        Length of scale bar in base pairs
    height : float, default 0.1
        Track height in figure units
    name : str, optional
        Track name
    label : str, optional
        Track label

    Returns
    -------
    callable
        Scale bar plotting function
    """

    def _plot(ax, *, dataset, chrom, start, end, idx, check_len, fig):

        end = start[-1]

        ax.add_patch(
            plt.Rectangle((end - length, 0), end, 0.3, color="black", linewidth=0.0)
        )

        ax.text(
            end - length / 2,
            0.45,
            f"{int(length//1000)} kb",
            fontsize=7,
            ha="center",
            va="bottom",
            color="black",
        )

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set(
            yticks=[],
            xticks=[],
        )

        return ax

    _plot.height = height
    _plot.track_name = name
    return _plot


def xaxis_plot(height=0.1, name=None):
    """
    Create a track showing genomic coordinates on x-axis.

    Parameters
    ----------
    height : float, default 0.1
        Track height in figure units
    name : str, optional
        Track name

    Returns
    -------
    callable
        X-axis plotting function
    """

    def _plot(ax, *, dataset, chrom, start, end, idx, check_len, fig):
        ax.tick_params(
            axis="x",
            top=True,
            labeltop=True,
            bottom=False,
            labelbottom=False,
            labelsize=7,
        )
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set(yticks=[])
        return ax

    _plot.height = height
    _plot.track_name = name
    return _plot


def spacer(height=0.1, name=None):
    """
    Create an empty spacer track.

    Parameters
    ----------
    height : float, default 0.1
        Spacer height in figure units
    name : str, optional
        Track name

    Returns
    -------
    callable
        Spacer plotting function
    """

    def _plot(ax, *args, **kwargs):
        ax.axis("off")
        return ax

    _plot.height = height
    _plot.track_name = name
    return _plot


def _check_flat_input(vals, check_len):
    if isinstance(vals, (tuple, list)):
        vals = np.array(vals)

    if not len(vals) == check_len:
        raise ValueError(
            "vals and must have one entry per region in the provided dataset."
        )

    return vals.squeeze()


def _name_or_none(x):
    return str(x.name).replace("_", " ") if hasattr(x, "name") else None


@cache
def _read_ideogram(cytobands_file):
    return read_csv(
        cytobands_file, 
        sep='\t', 
        header=None, 
        names=['chrom','start','end','band','stain']
    )


@borrow_kwargs(plt.Rectangle)
def ideogram(cytobands_file, height=0.15, name=None, **kwargs):
    """
    Create an ideogram track showing chromosome bands.

    Parameters
    ----------
    cytobands : pandas.DataFrame
        DataFrame with columns 'chrom', 'start', 'end', 'band', 'stain'
    height : float, default 0.15
        Track height in figure units
    name : str, optional
        Track name
    **kwargs
        Additional arguments passed to plt.Rectangle

    Returns
    -------
    callable
        Ideogram plotting function
    """

    color_lookup = {
        "gneg": "lightgrey",
        "gpos25": (0.6, 0.6, 0.6),
        "gpos50": (0.4, 0.4, 0.4),
        "gpos75": (0.2, 0.2, 0.2),
        "gpos100": (0.0, 0.0, 0.0),
        "acen": (0.8, 0.4, 0.4),
        "gvar": (0.8, 0.8, 0.8),
        "stalk": (0.9, 0.9, 0.9),
    }

    cytobands = _read_ideogram(cytobands_file)

    def _plot(ax, *, chrom, **kw):

        cytobands_df = cytobands[cytobands.chrom == chrom]

        for _start, _end, _diecolor in zip(
            cytobands_df.start, cytobands_df.end, cytobands_df.stain
        ):
            ax.add_patch(
                plt.Rectangle(
                    (_start, 0),
                    _end - _start,
                    1,
                    color=color_lookup[_diecolor],
                    **kwargs,
                )
            )

        for spine in ax.spines.values():
            spine.set(visible=True, linewidth=0.5, color="grey")

        ax.set(
            yticks=[],
            ylim=(0, 1),
            xticks=[],
        )
        return ax

    _plot.height = height
    _plot.track_name = name
    return _plot


@borrow_kwargs(plt.plot)
def line_plot(
    accessor,
    label=None,
    height=1,
    yticks=False,
    fill=False,
    name=None,
    color="#acacacff",
    ax_fn=lambda x: x,
    **kwargs,
):
    """
    Create a line plot track for continuous genomic data.

    Parameters
    ----------
    accessor : callable
        Function to extract data from dataset
    label : str, optional
        Track label
    height : float, default 1
        Track height in figure units
    yticks : bool, default False
        Whether to show y-axis ticks
    fill : bool, default False
        Whether to fill area under the line
    name : str, optional
        Track name for referencing
    color : str, default "#acacacff"
        Line color
    ax_fn : callable
        Function for additional axis customization
    **kwargs
        Additional arguments passed to plt.plot

    Returns
    -------
    callable
        Line plot function
    """

    def _plot(ax, *, dataset, chrom, start, end, idx, check_len, fig):

        vals = _check_flat_input(accessor(dataset), check_len)

        x = np.empty((start.size + end.size,), dtype=start.dtype)
        x[0::2] = start
        x[1::2] = end
        ax.plot(
            x,
            np.repeat(vals[idx], 2),
            color=color,
            label=label or _name_or_none(vals),
            **kwargs,
        )

        if fill:
            ax.fill_between(
                x, np.repeat(vals[idx], 2), color=color, **kwargs, alpha=0.2
            )

        _clean_ax(ax, axlabel=label or _name_or_none(vals), yticks=yticks)
        return ax_fn(ax)

    _plot.height = height
    _plot.track_name = name
    return _plot


@borrow_kwargs(plt.fill_between)
def fill_plot(
    accessor,
    label=None,
    height=1,
    yticks=False,
    name=None,
    color="#acacacff",
    ax_fn=lambda x: x,
    **kwargs,
):
    """
    Create a filled area plot track.

    Parameters
    ----------
    accessor : callable
        Function to extract data from dataset
    label : str, optional
        Track label
    height : float, default 1
        Track height in figure units
    yticks : bool, default False
        Whether to show y-axis ticks
    name : str, optional
        Track name
    color : str, default "#acacacff"
        Fill color
    ax_fn : callable
        Function for additional axis customization
    **kwargs
        Additional arguments passed to plt.fill_between

    Returns
    -------
    callable
        Fill plot function
    """

    def _plot(ax, *, dataset, chrom, start, end, idx, check_len, fig):

        vals = _check_flat_input(accessor(dataset), check_len)

        x = np.empty((start.size + end.size,), dtype=start.dtype)
        x[0::2] = start
        x[1::2] = end
        ax.fill_between(
            x,
            np.repeat(vals[idx], 2),
            color=color,
            label=label or _name_or_none(vals),
            **kwargs,
        )
        _clean_ax(ax, axlabel=label or _name_or_none(vals), yticks=yticks)
        ax.set_ylim(bottom=min(vals) * 0.97)
        return ax_fn(ax)

    _plot.height = height
    _plot.track_name = name
    return _plot


@borrow_kwargs(plt.bar)
def bar_plot(
    accessor,
    label=None,
    height=1,
    yticks=False,
    name=None,
    ax_fn=lambda x: x,
    color="#acacacff",  # Ensure color is used below or remove this line if unnecessary
    **kwargs,
):
    """
    Create a bar plot track for discrete genomic data.

    Parameters
    ----------
    accessor : callable
        Function to extract data from dataset
    label : str, optional
        Track label
    height : float, default 1
        Track height in figure units
    yticks : bool, default False
        Whether to show y-axis ticks
    name : str, optional
        Track name
    ax_fn : callable
        Function for additional axis customization
    color : str or array-like, default "#acacacff"
        Bar color(s)
    **kwargs
        Additional arguments passed to plt.bar

    Returns
    -------
    callable
        Bar plot function
    """

    def _plot(ax, *, dataset, chrom, start, end, idx, check_len, fig):

        vals = _check_flat_input(accessor(dataset), check_len)

        center = start + (end - start) / 2
        width = end - start

        ax.bar(
            center,
            vals[idx],
            width=width,
            color=(
                _check_flat_input(color)[idx]
                if not color is None and not isinstance(color, str)
                else color
            ),
            label=label or _name_or_none(vals),
            **kwargs,
        )
        _clean_ax(ax, axlabel=label or _name_or_none(vals), yticks=yticks)
        return ax_fn(ax)

    _plot.height = height
    _plot.track_name = name
    return _plot


@borrow_kwargs(plt.scatter)
def scatterplot(
    accessor,
    label=None,
    height=1,
    yticks=False,
    c=None,
    name=None,
    ax_fn=lambda x: x,
    **kwargs,
):
    """
    Create a scatter plot track for point data.

    Parameters
    ----------
    accessor : callable
        Function to extract data from dataset
    label : str, optional
        Track label
    height : float, default 1
        Track height in figure units
    yticks : bool, default False
        Whether to show y-axis ticks
    c : str or array-like, optional
        Point colors
    name : str, optional
        Track name
    ax_fn : callable
        Function for additional axis customization
    **kwargs
        Additional arguments passed to plt.scatter

    Returns
    -------
    callable
        Scatter plot function
    """

    def _plot(ax, *, dataset, chrom, start, end, idx, check_len, fig):

        vals = _check_flat_input(accessor(dataset), check_len)

        ax.scatter(
            start + (end - start) / 2,
            vals[idx],
            c=(
                _check_flat_input(c)[idx]
                if not c is None and not isinstance(c, str)
                else c
            ),
            label=label or _name_or_none(vals),
            **kwargs,
        )

        _clean_ax(ax, axlabel=label or _name_or_none(vals), yticks=yticks)
        return ax_fn(ax)

    _plot.height = height
    _plot.track_name = name
    return _plot


@borrow_kwargs(plt.pcolormesh)
def heatmap_plot(
    accessor,
    palette=diverging_palette,
    label=None,
    yticks=True,
    height=1,
    row_cluster=False,
    cluster_kw={},
    name=None,
    cbar=True,
    cbar_kw={},
    ax_fn=lambda x: x,
    **kwargs,
):
    """
    Create a heatmap track for matrix data.

    Parameters
    ----------
    accessor : callable
        Function to extract 2D data from dataset
    palette : str or colormap, default diverging_palette
        Color palette for heatmap
    label : str, optional
        Track label
    yticks : bool, default True
        Whether to show y-axis tick labels
    height : float, default 1
        Track height in figure units
    row_cluster : bool, default False
        Whether to cluster rows hierarchically
    cluster_kw : dict, default {}
        Keyword arguments for clustering
    name : str, optional
        Track name
    cbar : bool, default True
        Whether to show colorbar
    cbar_kw : dict, default {}
        Keyword arguments for colorbar
    ax_fn : callable
        Function for additional axis customization
    **kwargs
        Additional arguments passed to plt.pcolormesh

    Returns
    -------
    callable
        Heatmap plotting function
    """

    def _check_size(expected, x):
        if not len(x.shape) == 2:
            raise ValueError("The passed matrix must be two-dimensional.")

        if not x.shape[1] == expected:
            x = x.T

        if not x.shape[1] == expected:
            raise ValueError(
                "The passed matrix must have the same number of columns as there are regions in the dataset"
            )

        return x

    def _plot(ax, *, dataset, chrom, start, end, idx, check_len, fig):

        matrix = accessor(dataset)
        axlabel = label or _name_or_none(matrix)

        if not isinstance(matrix, (np.ndarray, DataArray)):
            raise TypeError(
                "The passed matrix must be a numpy array, or xarray DataArray."
            )

        matrix = _check_size(check_len, matrix)

        row_labels = []
        if isinstance(matrix, DataArray):
            dim = next((d for d in matrix.dims if d != "locus"), None)
            if not dim is None:
                row_labels = matrix.coords[dim].values
            matrix = matrix.values

        if row_cluster:
            row_order = _get_optimal_row_order(matrix, **cluster_kw)
            matrix = matrix[row_order]
            row_labels = [row_labels[i] for i in row_order]

        nrows = matrix.shape[0]
        x = np.arange(nrows + 1)
        y = np.concatenate([[start[0]], end])
        z = matrix[::-1, idx]

        im = ax.pcolormesh(y, x, z, cmap=palette, **kwargs)
        ax.set_yticks(np.arange(nrows) + 0.5)
        ax.set_yticklabels(row_labels[::-1], fontsize=7)
        ax.set(xticks=[])
        if not yticks:
            ax.set_yticks([])

        _set_axlabel(ax, axlabel)

        if cbar:

            box = ax.get_position()

            pad = cbar_kw.pop("pad", 0.035)
            width = cbar_kw.pop("width", 0.01)
            shrink = cbar_kw.pop("shrink", min(1, 1 / height))

            cax = fig.add_axes(
                [
                    box.xmax + pad,
                    box.ymin + (1 - shrink) * box.height / 2,
                    width,
                    shrink * box.height,
                ]
            )
            bar = fig.colorbar(
                im,
                cax=cax,
                orientation="vertical",
                **cbar_kw,
            )
            bar.set_ticks([bar.vmin, bar.vmax])
            bar.set_ticklabels(["min", "max"])
            bar.ax.tick_params(labelsize=7)

        return ax_fn(ax)

    _plot.height = height
    _plot.track_name = name
    return _plot


@borrow_kwargs(plt.Rectangle)
def categorical_plot(
    accessor,
    order=None,
    label=None,
    height=1,
    palette="tab10",
    edgecolor=None,
    name=None,
    ax_fn=lambda x: x,
    **kwargs,
):
    """
    Create a categorical data track with colored rectangles.

    Parameters
    ----------
    accessor : callable
        Function to extract categorical data from dataset
    order : list, optional
        Order of categories for display
    label : str, optional
        Track label
    height : float, default 1
        Track height in figure units
    palette : str or list, default "tab10"
        Color palette name or list of colors
    edgecolor : str, optional
        Edge color for rectangles
    name : str, optional
        Track name
    ax_fn : callable
        Function for additional axis customization
    **kwargs
        Additional arguments passed to plt.Rectangle

    Returns
    -------
    callable
        Categorical plot function
    """

    def _plot(ax, *, dataset, chrom, start, end, idx, check_len, fig):

        vals = _check_flat_input(accessor(dataset), check_len)

        categories = order or (
            vals.attrs["classes"][1:]
            if hasattr(vals, "attrs") and "classes" in vals.attrs
            else list(sorted(set(vals)))
        )

        if isinstance(palette, str):
            cmap = plt.get_cmap(palette)(np.linspace(0.1, 0.9, len(categories)))
        elif not len(palette) == len(categories):
            raise ValueError(
                "Palette must have the same number of colors as there are categories."
            )
        else:
            cmap = palette

        for _start, _end, _val in zip(start, end, vals[idx]):
            if _val in categories:
                try:
                    j = categories.index(_val)
                except ValueError:
                    continue

                ax.add_patch(
                    plt.Rectangle(
                        (_start, 0.1 * j),
                        _end - _start,
                        0.1,
                        color=cmap[j],
                        edgecolor=edgecolor,
                        **kwargs,
                    )
                )

        ax.set(
            yticks=list(0.1 * np.arange(len(categories)) + 0.05),
            yticklabels=[str(c) for c in categories],
            xticks=[],
            ylim=(0, 0.1 * len(categories)),
        )
        for spine in ax.spines.values():
            spine.set_visible(False)

        if not label is None:
            _set_axlabel(ax, label or _name_or_none(vals))
        return ax_fn(ax)

    _plot.height = height
    _plot.track_name = name
    return _plot


def custom_plot(fn, height=1, name=None):
    """
    Create a custom plotting track from a user-defined function.

    Parameters
    ----------
    fn : callable
        Custom plotting function that takes an axes object
    height : float, default 1
        Track height in figure units
    name : str, optional
        Track name

    Returns
    -------
    callable
        Custom plot function
    """

    def _plot(ax, *args, **kwargs):
        return fn(ax)

    _plot.height = height
    _plot.track_name = name
    return _plot


@cache
def _load_track_data(
    track_type,
    file,
    **properties,
):
    """
    Load track data using pygenometracks.

    Parameters
    ----------
    track_type : str
        Type of track (e.g., 'GtfTrack', 'BedTrack')
    file : str
        Path to data file
    **properties
        Additional track properties

    Returns
    -------
    Track object from pygenometracks

    Raises
    ------
    ImportError
        If pygenometracks is not installed
    """

    try:
        from pygenometracks import tracks
    except ImportError:
        raise ImportError(
            "pygenometracks is required to plot static tracks. "
            "Please install it using `pip install pygenometracks`."
        )

    logger.info(f"Loading track data from {file} ...")

    return getattr(tracks, track_type)(
        {
            "file": file,
            **properties,
        }
    )


def static_track(
    track_type,
    file,
    height=1,
    name=None,
    label=None,
    yticks=False,
    ax_fn=lambda x: x,
    **properties,
):
    """
    Create a static track using pygenometracks.

    Parameters
    ----------
    track_type : str
        Type of track (e.g., 'GtfTrack', 'BedTrack', 'BigwigTrack')
    file : str
        Path to the data file
    height : float, default 1
        Track height in figure units
    name : str, optional
        Track name
    label : str, optional
        Track label
    yticks : bool, default False
        Whether to show y-axis ticks
    ax_fn : callable
        Function for additional axis customization
    **properties
        Additional properties passed to the track

    Returns
    -------
    callable
        Static track plotting function
    """

    track = _load_track_data(
        track_type,
        file,
        **properties,
    )

    def _plot(ax, *, dataset, chrom, start, end, idx, check_len, fig):

        track.plot(
            ax,
            chrom,
            min(start),
            max(end),
        )

        _clean_ax(ax, axlabel=label, yticks=yticks)
        return ax_fn(ax)

    _plot.height = height
    _plot.track_name = name
    return _plot
