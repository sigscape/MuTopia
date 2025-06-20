from .track_plot import static_track


def gene_track(
    gtf,
    label="Genes",
    all_labels_inside=False,
    style="flybase",
    fontsize=5,
    ax_fn=lambda ax: ax.spines["bottom"].set_visible(False),
    **kw,
):
    """
    Create a gene annotation track from GTF file.

    Parameters
    ----------
    gtf : str
        Path to GTF file
    label : str, default "Genes"
        Track label
    all_labels_inside : bool, default False
        Whether to show all gene labels inside
    style : str, default "flybase"
        Gene track style
    fontsize : int, default 5
        Font size for labels
    ax_fn : callable
        Function to customize axes appearance
    **kw
        Additional keyword arguments

    Returns
    -------
    callable
        Track plotting function
    """

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
