import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Iterable
from functools import partial
from xarray import DataArray
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from numpy._core._multiarray_umath import _array_converter
from functools import cache
from ..genome_utils.bed12_utils import unstack_regions
from ..utils import borrow_kwargs, logger, diverging_palette, categorical_palette
from ..gtensor.gtensor import slice_regions, get_regions_filename

plt.rc("axes", linewidth=0.75)


def passthrough(data):
    def _passthrough(*args, **kwargs):
        return data

    return _passthrough


def pipeline(*fns):

    def _pipeline(data):
        for fn in fns:
            data = fn(data)
        return data

    return _pipeline


def acc(var_name, **sel):
    def _accessor(dataset):
        return dataset[var_name].sel(**sel).transpose(..., "locus")

    return _accessor


def feature_matrix(*feature_names):
    """
    Accessor function to retrieve multiple features from a dataset.

    Parameters:
    - feature_names: Names of the features to access.

    Returns:
    A function that retrieves the specified features from the dataset.
    """

    if len(feature_names) == 1 and isinstance(feature_names[0], Iterable):
        feature_names = list(feature_names[0])

    def _accessor(dataset):

        fnames = (
            list(feature_names)
            if len(feature_names) > 0
            else [
                name
                for name, arr in dataset.sections["Features"].items()
                if np.issubdtype(arr.dtype, np.number)
            ]
        )

        feature_matrix = DataArray(
            np.vstack([dataset.sections["Features"][name].values for name in fnames]),
            dims=("feature", "locus"),
            coords={"feature": fnames, "locus": dataset.coords["locus"].values},
            name="Features",
        )

        return feature_matrix.squeeze()

    return _accessor


def _xarr_op(fn):
    def run_fn(x):
        conv = _array_converter(x)
        out = fn(x)
        return conv.wrap(out)

    return run_fn


def _moving_average(bin_width, arr, alpha=10):

    if bin_width is None:
        weights = np.ones(alpha) / alpha
        ema = np.convolve(arr, weights, mode="same")
    else:
        # Fix moving average rate to weighted average rate to use sum(bin width * rate)/ (total bin width)
        window = np.ones(alpha)
        weighted_sum = np.convolve(arr * bin_width, window, mode="same")
        total_weight = np.convolve(bin_width, window, mode="same")

        # Compute the weighted moving average
        ema = weighted_sum / total_weight

    return ema


def clip(min_quantile=0.0, max_quantile=1.0):
    def _clip(arr):
        return np.clip(
            arr, np.nanquantile(arr, min_quantile), np.nanquantile(arr, max_quantile)
        )

    return _xarr_op(_clip)


def renorm(x):
    return x / np.nansum(x)


def apply_rows(fn):
    return partial(np.apply_along_axis, fn, 1)


def _get_optimal_row_order(data, **kwargs):
    return leaves_list(optimal_leaf_ordering(linkage(data, **kwargs), data))


def gene_track(
    gtf,
    label="Genes",
    all_labels_inside=False,
    style="flybase",
    fontsize=5,
    ax_fn=lambda ax: ax.spines["bottom"].set_visible(False),
    **kw,
):
    return static_track(
        "GtfTrack",
        gtf,
        label=label,
        all_labels_inside=all_labels_inside,
        style=style,
        fontsize=fontsize,
        ax_fn=ax_fn,
        **kw,
    )


class _GenomeView:

    def __init__(
        self,
        *,
        dataset,
        chrom,
        start,
        end,
        plot_kw,
    ):
        self.dataset = dataset
        self.chrom = chrom
        self.start = start
        self.end = end
        self._plot_kw = plot_kw

    def __call__(self, configuration, width=7):
        return self._plot(configuration(self), width=width)

    def _plot(self, plotting_fns, width=7):

        track_names = [
            fn.track_name for fn in plotting_fns if not fn.track_name is None
        ]
        if not len(track_names) == len(set(track_names)):
            raise ValueError(
                "If supplied, track names must be unique. "
                f"Found duplicates in {track_names}"
            )

        fig, ax = plt.subplots(
            len(plotting_fns),
            1,
            figsize=(width, sum(track.height for track in plotting_fns)),
            height_ratios=[track.height for track in plotting_fns],
            gridspec_kw={"hspace": 0.1},
            sharex=True,
        )

        if len(plotting_fns) == 1:
            ax = [ax]

        axs = {}

        kw = {
            "dataset": self.dataset,
            "chrom": self.chrom,
            "fig": fig,
            **self._plot_kw,
        }

        for i, (_ax, fn) in enumerate(zip(ax, plotting_fns)):
            axs[fn.track_name if not fn.track_name is None else i] = fn(_ax, **kw)
            _ax.set(xlim=(self.start, self.end))

        ax[0].set_title(f"{self.chrom}:{self.start}-{self.end}", fontsize=9, loc="left")
        return axs

    def smooth(self, alpha=10):
        smooth_fn = partial(
            _moving_average,
            self.dataset.sections["Regions"].length.values,
            alpha=alpha,
        )

        return _xarr_op(smooth_fn)


def make_view(dataset, *, chrom, start, end):

    dataset = slice_regions(dataset, chrom, start, end)
    n_regions = dataset.coords["locus"].size

    _, starts, ends, idxs = unstack_regions(
        dataset.coords["locus"].values,
        get_regions_filename(dataset),
        np.arange(n_regions),
    )

    plot_kw = dict(
        start=starts,
        end=ends,
        idx=idxs,
        check_len=n_regions,
    )

    return _GenomeView(
        dataset=dataset,
        chrom=chrom,
        start=start,
        end=end,
        plot_kw=plot_kw,
    )


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


@borrow_kwargs(plt.Rectangle)
def ideogram(cytobands, height=0.15, name=None, **kwargs):

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
