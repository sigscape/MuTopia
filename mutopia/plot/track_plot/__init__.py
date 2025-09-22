"""
Genome track plotting utilities for visualizing genomic data.

This module provides functions for creating genome browser-style visualizations
with multiple data tracks including line plots, heatmaps, and genomic annotations.

To plot a genome view from a G-Tensor object, first define a view, and a configuration.
A configuration is a Python function which describes the layout of the plot, but does
not reference any specific data. For example:

.. code-block:: python

    import mutopia.plot.track_plot as tr

    view = tr.make_view(dataset, region="chr1:100000-200000", title="Example Region")

    config = lambda view: (
        tr.scale_bar(length=1e7, height=0.1, scale="mb"),
        tr.ideogram(cytobands_file="path/to/cytobands.txt"),
        tr.line_plot(
            tr.select("Features/H3K9me3"),
            label="H3K9me3",
        )
    )

    ax = tr.plot_view(config, view)

You can create multi-column layouts and overlayed plots using ``tr.columns(...)`` and
``tr.stack_plots(...)``, respectively:

.. code-block:: python

    import mutopia.plot.track_plot as tr

    view = tr.make_view(dataset, region="chr1:100000-200000", title="Example Region")

    config = lambda view: (
        tr.columns(
            ...,  # Ellipsis to leave a blank column
            tr.line_plot(tr.select("Regions/exposures"), label="Exposure"),
        ),
        tr.columns(
            tr.stack_plots(
                tr.line_plot(tr.select("Regions/length"), label="Length"),
                tr.fill_plot(tr.select("Regions/exposures"), label="Exposure"),
                label="Overlayed Tracks",
            ),
            ...,
        ),
    )

    ax = tr.plot_view(config, view)

This way, the configuration can be reused for different datasets or regions of the genome.
"""

from .track_plot import *
from .transforms import *
import mutopia.plot.track_plot.tracks