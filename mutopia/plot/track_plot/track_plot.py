from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from xarray import DataArray, Dataset
from functools import cache
from dataclasses import dataclass, asdict
from typing import Any, Callable, Optional, Iterable, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.colors import Colormap
    from matplotlib.gridspec import GridSpec
    from mutopia.gtensor.gtensor import GTensorDataset

from mutopia.utils import logger, parse_region
from mutopia.palettes import diverging_palette

from .transforms import (
    _moving_average,
    _xarr_op,
    _get_optimal_row_order,
)

plt.rc("axes", linewidth=0.75)


__all__ = [
    "make_view",
    "plot_view",
    "columns",
    "stack_plots",
    "scale_bar",
    "xaxis_plot",
    "spacer",
    "ideogram",
    "line_plot",
    "fill_plot",
    "bar_plot",
    "scatterplot",
    "static_track",
    "heatmap_plot",
    "categorical_plot",
    "custom_plot",
    "text_banner",
    "center_at_zero",
]


def center_at_zero(ax: "Axes") -> "Axes":
    """
    Center the y-axis of the plot at zero.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to modify.
    
    Returns
    -------
    matplotlib.axes.Axes
    """
    ax.spines["bottom"].set_visible(False)
    ax.axhline(0, color="k", linewidth=0.75)
    return ax


@dataclass
class GenomeView:
    chrom: str
    interval: tuple[int, int]
    dataset: Dataset
    title: Optional[str]
    n_regions: int
    start: np.ndarray
    end: np.ndarray
    idx: np.ndarray

    def smooth(self, alpha: float = 10) -> Callable[[DataArray], DataArray]:
        smooth_fn = partial(
            _moving_average,
            self.dataset.sections["Regions"].length.values,
            alpha=alpha,
        )

        return _xarr_op(smooth_fn)

    def renorm(self, x: np.ndarray) -> np.ndarray:
        return x / np.nansum(x) * self.n_regions


def make_view(
    dataset: "GTensorDataset",
    region: Optional[str] = None,
    title: Optional[str] = None,
) -> GenomeView:
    """
    Create a GenomeView for a specific genomic region.

    Parameters
    ----------
    dataset : GTensorDataset
        Input genomic dataset (or interface) containing Regions coordinates.
    region : str, optional
        Region spec like "chr1:100000-200000". If None and dataset has a single
        chromosome, the full range for that chromosome is used.
    title : str, optional
        Optional figure title used by plot_view.

    Returns
    -------
    GenomeView
        Configured genome view object.

    Examples
    --------
    >>> import mutopia.plot.track_plot as tr
    >>> view = tr.make_view(ds, region="chr1:1_000_000-1_200_000", title="chr1 slice")
    """

    from mutopia.genome_utils.bed12_utils import unstack_regions
    from mutopia.gtensor.gtensor import slice_regions, get_regions_filename

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

    start = min(starts)
    end = max(ends)

    return GenomeView(
        chrom=chrom,
        interval=(start, end),
        dataset=dataset,
        start=starts,
        end=ends,
        idx=idxs,
        n_regions=n_regions,
        title=title,
    )


def plot_view(
    configuration: Callable[..., Any],
    view: GenomeView,
    *args: Any,
    width: float = 7,
    gridpsec_kw: dict = {"hspace": 0.1, "wspace": 0.1},
    width_ratios: Optional[Sequence[float]] = None,
    **kwargs: Any,
) -> "Figure":
    """
    Render a genome view using a configuration of track functions.

    Parameters
    ----------
    configuration : callable
        A function that takes the GenomeView and returns one or more track
        callables (e.g., line_plot(...), heatmap_plot(...)).
    view : GenomeView
        The genome view produced by make_view.
    width : float, default 7
        Figure width in inches.
    gridpsec_kw : dict, optional
        GridSpec kwargs for inter-row/column spacing.
    width_ratios : sequence of float, optional
        Relative column widths for multi-column layouts.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.

    Examples
    --------
    >>> import mutopia.plot.track_plot as tr
    >>> view = tr.make_view(ds, region="chr1:1_000_000-1_100_000")
    >>> cfg = lambda v: tr.line_plot(tr.select("Regions/length"), label="Length")
    >>> fig = tr.plot_view(cfg, view)
    """

    def list_if_not(x):
        return x if isinstance(x, Iterable) else [x]

    def get_n_cols(track):
        return getattr(track, "num_columns", 1)

    tracks = list_if_not(configuration(view, *args, **kwargs))

    tracks = [track_fn for track in tracks for track_fn in list_if_not(track)]

    n_rows = len(tracks)
    n_cols = max(get_n_cols(track) for track in tracks)

    if not all([get_n_cols(track) == n_cols for track in tracks]):
        raise ValueError(
            f"All tracks must have the same number of columns. Found: {[get_n_cols(track) for track in tracks]}"
        )

    track_names = [fn.track_name for fn in tracks if not fn.track_name is None]

    if not len(track_names) == len(set(track_names)):
        raise ValueError(
            "If supplied, track names must be unique. "
            f"Found duplicates in {track_names}"
        )

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(width, sum(track.height for track in tracks)),
        gridspec_kw=gridpsec_kw,
        height_ratios=[track.height for track in tracks],
        width_ratios=width_ratios,
        squeeze=False,
        sharex="col",
    )
    axes = list_if_not(axes)

    for row_num, track in enumerate(tracks):
        ax = axes[row_num, 0] if n_cols == 1 else axes[row_num, :]
        track(
            ax,
            fig=fig,
            **asdict(view),
        )

    if hasattr(view, "title") and view.title:
        fig.suptitle(view.title, fontsize=9, ha="left", x=0.02)
    return fig


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
    *plotting_fns: Callable[..., Any],
    height: float = 1,
    label: Optional[str] = None,
    legend: bool = True,
    name: Optional[str] = None,
    ax_fn: Callable[["Axes"], "Axes"] = lambda x: x,
) -> Callable[..., "Axes"]:
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
        Combined plotting function.

    Examples
    --------
    >>> import mutopia.plot.track_plot as tr
    >>> cfg = lambda v: tr.stack_plots(
    ...     tr.line_plot(tr.select("Regions/length"), label="Length"),
    ...     tr.line_plot(tr.select("Regions/exposures"), label="Exposure"),
    ...     label="Tracks"
    ... )
    >>> fig = tr.plot_view(cfg, tr.make_view(ds, region="chr1:1-1_000_000"))
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


def columns(
    *plotting_fns: Callable[..., Any],
    height: float = 1,
    name: Optional[str] = None,
) -> Callable[..., Any]:
    """
    Create a multi-column track container.

    Use this to lay out multiple plotting tracks side-by-side within a
    single row. Pass plotting callables for each column. Use ``...``
    (Ellipsis) to leave a blank column in the layout.

    Parameters
    ----------
    *plotting_fns : callable | Ellipsis
        Plotting callables to place in each column (e.g.,
        ``line_plot(...)``, ``heatmap_plot(...)``). Use ``...`` to skip a column.
    height : float, default 1
        Track row height in figure units.
    name : str, optional
        Optional track name for referencing.

    Returns
    -------
    callable
        A plotting callable that receives an array of Axes for the row and
        draws each column.

    Examples
    --------
    >>> import mutopia.plot.track_plot as tr
    >>> view = tr.make_view(ds, region="chr1:1_000_000-1_100_000")
    >>> cfg = lambda v: tr.columns(
    ...     tr.line_plot(tr.select("Regions/length"), label="Length"),
    ...     ...,  # leave middle column blank
    ...     tr.bar_plot(tr.select("Regions/exposures"), label="Exposure"),
    ... )
    >>> fig = tr.plot_view(cfg, view, width_ratios=[1, 0.6, 1.2])
    """

    def _plot(axes, *args, **kwargs):
        for i, plot_fn in enumerate(plotting_fns):
            if not plot_fn is Ellipsis:
                plot_fn(axes[i], *args, **kwargs)
            else:
                axes[i].axis("off")

    _plot.height = height
    _plot.track_name = name
    _plot.num_columns = len(plotting_fns)
    return _plot


def scale_bar(
    length: float = 10000,
    height: float = 0.1,
    name: Optional[str] = None,
    label: Optional[str] = None,
    scale: str = "kb",
) -> Callable[["Axes"], "Axes"]:
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
        Scale bar plotting function.

    Examples
    --------
    >>> import mutopia.plot.track_plot as tr
    >>> cfg = lambda v: tr.scale_bar(length=100_000, scale="kb")
    >>> fig = tr.plot_view(cfg, tr.make_view(ds, region="chr1:1-2_000_000"))
    """

    scale_dict = {
        "bp": 1,
        "kb": 1_000,
        "mb": 1_000_000,
    }

    if not scale in scale_dict:
        raise ValueError(
            f"Unknown scale: {scale}, must be one of {list(scale_dict.keys())}"
        )

    annotation = f"{int(length) // scale_dict[scale]:d} {scale.title()}"

    def _plot(ax, *, interval, start, end, **kw):

        end = start[-1]

        ax.add_patch(
            plt.Rectangle((end - length, 0), end, 0.3, color="black", linewidth=0.0)
        )

        ax.text(
            end - length / 2,
            0.45,
            annotation,
            fontsize=7,
            ha="center",
            va="bottom",
            color="black",
        )

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set(yticks=[], xticks=[], xlim=interval)

        return ax

    _plot.height = height
    _plot.track_name = name
    return _plot


def xaxis_plot(height: float = 0.1, name: Optional[str] = None) -> Callable[..., "Axes"]:
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
        X-axis plotting function.

    Examples
    --------
    >>> import mutopia.plot.track_plot as tr
    >>> cfg = lambda v: tr.xaxis_plot()
    >>> fig = tr.plot_view(cfg, tr.make_view(ds, region="chr1:1-2_000_000"))
    """

    def _plot(ax, *, interval, **kw):
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
        ax.set_xlim(interval)
        return ax

    _plot.height = height
    _plot.track_name = name
    return _plot


def spacer(height: float = 0.1, name: Optional[str] = None) -> Callable[..., "Axes"]:
    """
    Insert an empty spacer track to separate other tracks.

    Parameters
    ----------
    height : float, default 0.1
        Spacer height in figure units.
    name : str, optional
        Optional track name.

    Returns
    -------
    callable
        Plotting callable that renders an empty row.

    Examples
    --------
    >>> import mutopia.plot.track_plot as tr
    >>> cfg = lambda v: (tr.line_plot(tr.select("Regions/length")), tr.spacer(0.25))
    >>> _ = tr.plot_view(cfg, tr.make_view(ds, region="chr1:1-2_000_000"))
    """

    def _plot(ax, *args, **kwargs):
        ax.axis("off")
        return ax

    _plot.height = height
    _plot.track_name = name
    return _plot


def text_banner(label: str, height: float = 0.15, name: Optional[str] = None) -> Callable[..., "Axes"]:
    """
    Create a simple text banner track with a horizontal rule.

    Parameters
    ----------
    label : str
        Text to display centered across the track.
    height : float, default 0.15
        Track height in figure units.
    name : str, optional
        Optional track name.

    Returns
    -------
    callable
        Plotting callable for use inside a plot_view configuration.

    Examples
    --------
    >>> import mutopia.plot.track_plot as tr
    >>> cfg = lambda v: tr.text_banner("My Region")
    >>> _ = tr.plot_view(cfg, tr.make_view(ds, region="chr1:1-2_000_000"))
    """

    def _plot(ax, *, start, end, **_):
        start = start[0]
        end = end[-1]
        ax.text(
            0.5,
            0.1,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            transform=ax.transAxes,
        )
        ax.axhline(0.1, color="k", linewidth=0.5)
        ax.axis("off")
        return ax

    _plot.height = height
    _plot.track_name = name
    return _plot


def _check_flat_input(vals, n_regions):
    if isinstance(vals, (tuple, list)):
        vals = np.array(vals)

    if not len(vals) == n_regions:
        raise ValueError(
            "vals and must have one entry per region in the provided dataset."
        )

    return vals.squeeze()


def _name_or_none(x):
    return str(x.name).replace("_", " ") if hasattr(x, "name") else None


@cache
def _read_ideogram(cytobands_file):
    from pandas import read_csv

    return read_csv(
        cytobands_file,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "band", "stain"],
    )



def ideogram(
    cytobands_file: str,
    height: float = 0.15,
    name: Optional[str] = None,
    **kwargs: Any,
) -> Callable[..., "Axes"]:
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
        Ideogram plotting function.

    Examples
    --------
    >>> import mutopia.plot.track_plot as tr
    >>> cfg = lambda v: tr.ideogram("/path/to/cytoBand.txt.gz")
    >>> fig = tr.plot_view(cfg, tr.make_view(ds, region="chr1:1-5_000_000"))
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

    def _plot(ax, *, interval, chrom, **kw):

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
        ax.set_xlim(interval)
        return ax

    _plot.height = height
    _plot.track_name = name
    return _plot


def line_plot(
    accessor: Callable[[Dataset], np.ndarray | DataArray | Sequence[float]],
    label: Optional[str] = None,
    height: float = 1,
    yticks: bool = False,
    fill: bool = False,
    name: Optional[str] = None,
    color: str = "#acacacff",
    ax_fn: Callable[["Axes"], "Axes"] = lambda x: x,
    **kwargs: Any,
) -> Callable[..., "Axes"]:
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
        Line plot function.

    Examples
    --------
    >>> import mutopia.plot.track_plot as tr
    >>> cfg = lambda v: tr.line_plot(tr.select("Regions/length"), label="Length")
    >>> fig = tr.plot_view(cfg, tr.make_view(ds, region="chr1:1-1_000_000"))
    """

    def _plot(ax, *, dataset, start, end, idx, n_regions, **kw):

        vals = _check_flat_input(accessor(dataset), n_regions)

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
            ax.set_ylim(bottom=0.0)

        _clean_ax(ax, axlabel=label or _name_or_none(vals), yticks=yticks)
        return ax_fn(ax)

    _plot.height = height
    _plot.track_name = name
    return _plot


def fill_plot(
    accessor: Callable[[Dataset], np.ndarray | DataArray | Sequence[float]],
    label: Optional[str] = None,
    height: float = 1,
    yticks: bool = False,
    name: Optional[str] = None,
    color: str = "#acacacff",
    ax_fn: Callable[["Axes"], "Axes"] = lambda x: x,
    **kwargs: Any,
) -> Callable[..., "Axes"]:
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
        Fill plot function.

    Examples
    --------
    >>> import mutopia.plot.track_plot as tr
    >>> cfg = lambda v: tr.fill_plot(tr.select("Regions/length"), label="Length")
    >>> fig = tr.plot_view(cfg, tr.make_view(ds, region="chr1:1-1_000_000"))
    """

    def _plot(ax, *, dataset, start, end, idx, n_regions, **kw):

        vals = _check_flat_input(accessor(dataset), n_regions)

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


def bar_plot(
    accessor: Callable[[Dataset], np.ndarray | DataArray | Sequence[float]],
    label: Optional[str] = None,
    height: float = 1,
    yticks: bool = False,
    name: Optional[str] = None,
    ax_fn: Callable[["Axes"], "Axes"] = lambda x: x,
    color: str | Sequence[str] | None = "#acacacff",
    **kwargs: Any,
) -> Callable[..., "Axes"]:
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
        Bar plot function.

    Examples
    --------
    >>> import mutopia.plot.track_plot as tr
    >>> cfg = lambda v: tr.bar_plot(tr.select("Regions/length"), label="Length")
    >>> fig = tr.plot_view(cfg, tr.make_view(ds, region="chr1:1-1_000_000"))
    """

    def _plot(ax, *, dataset, start, end, idx, n_regions, **kw):

        vals = _check_flat_input(accessor(dataset), n_regions)

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


def scatterplot(
    accessor: Callable[[Dataset], np.ndarray | DataArray | Sequence[float]],
    label: Optional[str] = None,
    height: float = 1,
    yticks: bool = False,
    c: str | Sequence[str] | None = None,
    name: Optional[str] = None,
    ax_fn: Callable[["Axes"], "Axes"] = lambda x: x,
    **kwargs: Any,
) -> Callable[..., "Axes"]:
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
        Scatter plot function.

    Examples
    --------
    >>> import mutopia.plot.track_plot as tr
    >>> cfg = lambda v: tr.scatterplot(tr.select("Regions/length"), label="Length")
    >>> fig = tr.plot_view(cfg, tr.make_view(ds, region="chr1:1-1_000_000"))
    """

    def _plot(ax, *, dataset, start, end, idx, n_regions, **kw):

        vals = _check_flat_input(accessor(dataset), n_regions)

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


def heatmap_plot(
    accessor: Callable[[Dataset], np.ndarray | DataArray],
    palette: "Colormap" | str = diverging_palette,
    label: Optional[str] = None,
    yticks: bool = True,
    height: float = 1,
    row_cluster: bool = False,
    cluster_kw: dict = {},
    name: Optional[str] = None,
    cbar: bool = True,
    cbar_kw: dict = {},
    ax_fn: Callable[["Axes"], "Axes"] = lambda x: x,
    **kwargs: Any,
) -> Callable[..., "Axes"]:
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
        Heatmap plotting function.

    Examples
    --------
    >>> import mutopia.plot.track_plot as tr
    >>> cfg = lambda v: tr.heatmap_plot(tr.select("Some/Matrix"), label="Matrix")
    >>> fig = tr.plot_view(cfg, tr.make_view(ds, region="chr1:1-1_000_000"))
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

    def _plot(ax, *, dataset, start, end, idx, n_regions, fig, **kw):

        matrix = accessor(dataset)
        axlabel = label or _name_or_none(matrix)

        if not isinstance(matrix, (np.ndarray, DataArray)):
            raise TypeError(
                "The passed matrix must be a numpy array, or xarray DataArray."
            )

        matrix = _check_size(n_regions, matrix)

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


def categorical_plot(
    accessor: Callable[[Dataset], Sequence[Any] | DataArray | np.ndarray],
    order: Optional[Sequence[Any]] = None,
    label: Optional[str] = None,
    height: float = 1,
    palette: str | Sequence[str] = "tab10",
    edgecolor: Optional[str] = None,
    name: Optional[str] = None,
    ax_fn: Callable[["Axes"], "Axes"] = lambda x: x,
    **kwargs: Any,
) -> Callable[..., "Axes"]:
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
        Categorical plot function.

    Examples
    --------
    >>> import mutopia.plot.track_plot as tr
    >>> cfg = lambda v: tr.categorical_plot(tr.select("Regions/chrom"))
    >>> fig = tr.plot_view(cfg, tr.make_view(ds, region="chr1:1-1_000_000"))
    """

    def _plot(ax, *, dataset, start, end, idx, n_regions, **kw):

        vals = _check_flat_input(accessor(dataset), n_regions)

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


def custom_plot(
    fn: Callable[["Axes"], Any],
    height: float = 1,
    name: Optional[str] = None,
) -> Callable[..., "Axes"]:
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
        Custom plot function.
    """

    def _plot(ax, *args, **kwargs):
        return fn(ax)

    _plot.height = height
    _plot.track_name = name
    return _plot


@cache
def _load_track_data(
    track_type: str,
    file: str,
    **properties: Any,
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
    track_type: str,
    file: str,
    height: float = 1,
    name: Optional[str] = None,
    label: Optional[str] = None,
    yticks: bool = False,
    ax_fn: Callable[["Axes"], "Axes"] = lambda x: x,
    **properties: Any,
) -> Callable[..., "Axes"]:
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
        Static track plotting function.

    Examples
    --------
    >>> import mutopia.plot.track_plot as tr
    >>> cfg = lambda v: tr.static_track("BigWigTrack", "/path/to/signal.bw", title="Signal")
    >>> fig = tr.plot_view(cfg, tr.make_view(ds, region="chr1:1-1_000_000"))
    """

    track = _load_track_data(
        track_type,
        file,
        **properties,
    )

    def _plot(ax, *, chrom, start, end, **kw):

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
