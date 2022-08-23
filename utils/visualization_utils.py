import numpy as np
from matplotlib import pyplot as plt

def visualize_accuracy_across_time_bins(
    accuracies, 
    pre_interval, 
    post_interval, 
    interval_size, 
    ax,
    label=None,
    right_align=False,
):
    """Plots accuracies across time bins as a shaded line plot

    Args:
        accuracies: num_bins x num_runs np.array
        pre_interval: int, in miliseconds
        post_interval: int, in miliseconds
        interval_size: int, in miliseconds
    """
    means = np.nanmean(accuracies, axis=1)
    stds = np.nanstd(accuracies, axis=1)
    x = np.arange(-pre_interval, post_interval, interval_size)
    if right_align:
        # every x timepoint indicates the right of the bin
        x = x + interval_size
    ax.plot(x, means, label=label)
    ax.fill_between(x, means - stds, means + stds, alpha=0.5)


def plot_hist_of_selections(feature_selections, feature_dim, ax):
    dist = feature_selections[feature_dim]
    ax.hist(dist)