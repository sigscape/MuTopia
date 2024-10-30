import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import leaves_list, linkage

def _moving_average(arr, alpha, bin_width=None):
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


def plot_mutation_rate(
        rate, 
        bin_width=None,
        smoothing = 300,
        ax = None, 
        plot_ema = True, 
        plot_raw = True,
        ylim = None,
        normalize = True,
        add_grid = True,
        grid_spacing = 10000,
        figsize = (15,1.5),
        point_kw = {},
        ema_kw = {}
):
    """
    Plots the mutation rate.

    Parameters
    ----------
    rate : array-like
        The mutation rate values.
    bin_width : array-like, optional
        The widths of the bins corresponding to each rate value. If None, bin width weighted average operation will be disabled. Defaults to None.
    smoothing : int, optional
        The smoothing window size. Defaults to 300.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes will be created. Defaults to None.
    plot_ema : bool, optional
        Whether to plot the exponential moving average. Defaults to True.
    plot_raw : bool, optional
        Whether to plot the raw mutation rate. Defaults to True.
    ylim : tuple, optional
        The y-axis limits. Defaults to None.
    normalize : bool, optional
        Whether to normalize the mutation rate. Defaults to True.
    add_grid : bool, optional
        Whether to add grid lines. Defaults to True.
    grid_spacing : int, optional
        The spacing between grid lines. Defaults to 10000.
    figsize : tuple, optional
        The figure size. Defaults to (15, 1.5).
    point_kw : dict, optional
        Additional keyword arguments for the scatter plot. Defaults to {}.
    ema_kw : dict, optional
        Additional keyword arguments for the exponential moving average plot. Defaults to {}.

    Returns
    -------
    matplotlib.axes.Axes
        The plotted axes.
    """
    if bin_width is not None and  len(rate) != len(bin_width):
        raise ValueError("Rate and bin widths must have the same length")   
        
    if ax is None:
        _, ax = plt.subplots(1,1, figsize=figsize, sharex=True)
    
    if plot_raw:
        defaults = {
            's': 0.05,
            'alpha': 0.3,
            'color': 'black',
        }
        defaults.update(point_kw)

        ax.scatter(
            range(len(rate)),
            rate if not normalize else rate/rate.sum(),
            **defaults
        )

    if plot_ema:
        defaults = {
            'color': 'red',
            'linewidth': 0.5,
            'alpha' : 1.,
        }
        defaults.update(ema_kw)

        smoothed_rate = _moving_average(rate, smoothing, bin_width=bin_width)

        ax.plot(
            range(len(rate)),
            smoothed_rate if not normalize else smoothed_rate/smoothed_rate.sum(),
            **defaults
        )
        
    ax.set(
        xticks = [], 
        ylim = ylim,
        xlim = (0, len(rate)),
    )

    if add_grid:
        # Add vertical grid lines
        for i in range(0,len(rate), grid_spacing):
            ax.axvline(x=i, color='grey', linestyle='--', linewidth=0.5)

        if normalize:
            ax.axhline(y=1/len(rate), color='grey', linestyle='--', linewidth=0.5)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    return ax



def plot_rate_matrix(
        rates,
        feature_names, 
        mesoscale_features=[],
        bin_width=None,
        width=20, 
        height=0.5, 
        optimal_ordering=True, 
        ax=None,
        smoothing=250,
        ylim=None,
        grid_spacing=2500,
        normalize=True,
        **plot_kw,
    ):
    """
    Plot the mutation rate matrix.

    Parameters
    ----------
    rates : numpy.ndarray
        The mutation rate matrix.
    feature_names : list
        The names of the features.
    mesoscale_features : list, optional
        The names of the meso-scale features, this must be a subset of feature_names.
    bin_width : array-like, optional
        The widths of the bins corresponding to each rate value. Default is None.
    width : int, optional
        The width of the plot in inches. Default is 20.
    height : float, optional
        The height of each subplot in inches. Default is 0.5.
    optimal_ordering : bool, optional
        Whether to use optimal feature ordering. Default is True.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If not provided, a new figure and axes will be created.
    smoothing : int, optional
        The smoothing parameter for the mutation rate plot. Default is 250.
    ylim : tuple, optional
        The y-axis limits for the plot. Default is None.
    grid_spacing : int, optional
        The grid spacing for the mutation rate plot. Default is 2500.
    normalize : bool, optional
        Whether to normalize the mutation rates. Default is True.
    **plot_kw : keyword arguments, optional
        Additional keyword arguments to pass to the plot_mutation_rate function.

    Returns
    -------
    matplotlib.axes.Axes
        The plotted axes.
    """

    n_features, _ = rates.shape
    assert len(feature_names) == n_features

    if optimal_ordering:
        optimal_feature_ordering = leaves_list(
            linkage(
                rates, 
                method='average', 
                metric='cosine', 
                optimal_ordering=True
            )
        )
    else:
        optimal_feature_ordering = range(n_features)

    if ax is None:
        _, ax = plt.subplots(n_features, 1,
                             figsize=(width, height * n_features),
                             sharex=True
                            )

    plot_kw = dict(
        normalize=normalize, 
        smoothing=smoothing,
        ylim=ylim, 
        plot_raw=False, 
        ema_kw=plot_kw,
        grid_spacing=grid_spacing,
    )

    for f in optimal_feature_ordering:
        plot_mutation_rate(rates[f], bin_width=bin_width, **plot_kw, ax=ax[f])
        ax[f].set(yticks=[0], ylabel=feature_names[f])
        ax[f].yaxis.label.set(rotation='horizontal', ha='right', va='center')

    return ax



def plot_corpus_features(corpus, **kwargs):
    """
    Plot the corpus features.

    Parameters
    ----------
    corpus : Corpus
        The corpus object containing the X_matrix and feature_names.
    **kwargs : dict
        Additional keyword arguments to be passed to the plot_rate_matrix function.

    Returns
    -------
    matplotlib.figure.Figure
        The plot of the corpus features.
    """
    return plot_rate_matrix(
        corpus.X_matrix.T, 
        corpus.feature_names, 
        **kwargs
    )