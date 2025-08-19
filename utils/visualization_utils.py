import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from stl import mesh  # pip install numpy-stl
import tempfile
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from itertools import accumulate
import matplotlib.patches as patches
from scipy import stats
import utils.classifier_utils as classifier_utils
from spike_tools import (
    general as spike_general,
)
import utils.spike_utils as spike_utils
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
from scripts.anova_analysis.run_anova import load_data
from constants.behavioral_constants import *
from constants.decoding_constants import *
import copy
import os
from scipy.stats import ttest_ind
import itertools
import matplotlib.gridspec as gridspec
import utils.behavioral_utils as behavioral_utils
from matplotlib.colors import ListedColormap, BoundaryNorm


# REGION_TO_COLOR = {
#     "Amygdala": "#00ff00",
#     "Anterior Cingulate Gyrus": "#ff4500",
#     "Basal Ganglia": "#0000ff",
#     "Claustrum": "#ffd700",
#     "Hippocampus/MTL": "#00ffff",
#     "Parietal Cortex": "#191970",
#     "Prefrontal Cortex": "#bc8f8f",
#     "Premotor Cortex": "#006400",
#     "Visual Cortex": "#ff1493",
# }

REGION_TO_COLOR = {
    "AMY": "#30123b",
    "BG": "#3e9bfe",
    "ITC": "#46f884",
    "HPC": "#e1dd37",
    "LPFC": "#f05b12",
    "ACC": "#7a0403",
    "shuffle": "gray"
}

REGION_ORDER = ["AMY", "BG", "HPC", "ITC", "LPFC", "ACC", "shuffle"]

REGION_TO_DISPLAY_NAMES = {
    "amygdala_Amy": "AMY",
    "basal_ganglia_BG": "BG",
    "inferior_temporal_cortex_ITC": "ITC",
    "medial_pallium_MPal": "HPC",
    "lateral_prefrontal_cortex_lat_PFC": "LPFC",
    "anterior_cingulate_gyrus_ACgG": "ACC",
    "shuffle": "shuffle"
}

MODE_TO_COLOR = {
    "Color": "#1B9E77",
    "Shape": "#D95F02",
    "Pattern": "#7570B3",
    "preference": "tab:blue",
    "confidence": "tab:red",
    "reward": "tab:green",
    "feature selected": "tab:orange",
    "shuffle": "gray"
}

MODE_TO_DISPLAY_NAMES = {
    "Color": "Color",
    "Shape": "Shape",
    "Pattern": "Pattern",
    "pref": "preference",
    "conf": "confidence",
    "reward": "reward",
    "choice": "feature selected",
    "shuffle": "shuffle"
}

SIG_LEVELS = [
    (0.001, 6),  # p < 0.001 → thickest
    (0.01, 4),   # p < 0.01 → medium
    (0.05, 2),   # p < 0.05 → thin
]

ANOVA_SIG_LEVELS = [
    ("99th", 4),
    ("95th", 2),
]


def visualize_accuracy_across_time_bins(
    accuracies, 
    pre_interval, 
    post_interval, 
    interval_size, 
    ax,
    label=None,
    right_align=False,
    color=None,
    add_err=True, 
    sem=False,
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
    mean_line, = ax.plot(x, means, label=label, linewidth=2)
    if add_err:
        err = stds / np.sqrt(accuracies.shape[1]) if sem else stds
        std_line = ax.fill_between(x, means - err, means + err, alpha=0.5)
    if color:
        mean_line.set_color(color)
        if add_err:
            std_line.set_color(color)


def visualize_accuracy_bars(accuracies, labels, ax):
    sns.barplot(data=accuracies, capsize=.1, errorbar='sd', ax=ax)
    sns.swarmplot(data=accuracies, color="0", alpha=.35, ax=ax, size=0.5)
    ax.set_xticklabels(labels)



def plot_hist_of_selections(feature_selections, feature_dim, ax):
    dist = feature_selections[feature_dim]
    ax.hist(dist)


# def plot_values_by_trial(trial_numbers, )
#     """
#     Plots values by trial as a color grid, 
#     """

def plotly_add_glass_brain(fs, fig1, subject, areas=['brain'], show_axis=False):
    '''
    Adds a glass brain to a plotly figure
    Code for rendering stl file is from here:
    https://chart-studio.plotly.com/~empet/15276/converting-a-stl-mesh-to-plotly-gomes/#/
    
    Parameters
    ----------------
    fig1 : plotly figure
    subject : subject of whom's glass brain to plot
    areas : list of strings indicating areas to print. List must limited to
        'brain', 'fef', 'dlpfc', 'mpfc', 'hippocampus'
        Use areas='all' if you want to plot all areas
    show_axis : whether or not to include axis
    '''
    def stl2mesh3d(stl_mesh):
        # stl_mesh is read by nympy-stl from a stl file; it is  an array of faces/triangles 
        # (i.e. three 3d points) 
        # this function extracts the unique vertices and the lists I, J, K to define a Plotly mesh3d
        p, q, r = stl_mesh.vectors.shape #(p, 3, 3)
        # the array stl_mesh.vectors.reshape(p*q, r) can contain multiple copies of the same vertex;
        # extract unique vertices from all mesh triangles
        vertices, ixr = np.unique(stl_mesh.vectors.reshape(p*q, r), return_inverse=True, axis=0)
        I = np.take(ixr, [3*k for k in range(p)])
        J = np.take(ixr, [3*k+1 for k in range(p)])
        K = np.take(ixr, [3*k+2 for k in range(p)])
        return vertices, I, J, K
    
    fig_data = fig1.data
    
    if areas=='all':
        areas = ['brain', 'fef', 'dlpfc', 'mpfc', 'hippocampus']
    
    for area in areas:
        if area=='brain':
            filename = 'glass brain.stl'
            colorscale= [[0, 'whitesmoke'], [1, 'whitesmoke']]
        elif area=='fef':
            filename = 'FEF.stl'
            colorscale= [[0, 'darkblue'], [1, 'darkblue']]
        elif area=='dlpfc':
            filename = 'dlPFC.stl'
            colorscale= [[0, 'lightgreen'], [1, 'lightgreen']]
        elif area=='mpfc':
            filename = 'mPFC.stl'
            colorscale= [[0, 'darkgreen'], [1, 'darkgreen']]
        elif area=='hippocampus':
            filename = 'Hippocampal.stl'
            colorscale= [[0, 'coral'], [1, 'coral']]
            
        # stl_file = 'nhp-lfp/wcst-preprocessed/rawdata/sub-'+subject+'/anatomy/'+filename
        stl_file = '/data/rawdata/sub-'+subject+'/anatomy/'+filename
        # if not fs.exists(stl_file):
        #     print(stl_file + ' does not exist, not including...')
        #     continue
    
        # tmp = tempfile.NamedTemporaryFile()
        # with open(tmp.name, mode='wb') as f:
        #     fs.get_file(stl_file, tmp.name)
        my_mesh = mesh.Mesh.from_file(stl_file)
    
        vertices, I, J, K = stl2mesh3d(my_mesh)
        x, y, z = vertices.T
    
        mesh3D = go.Mesh3d(x=x, y=y, z=z, i=I, j=J, k=K, \
                           opacity=0.14, colorscale=colorscale, intensity=z, showscale=False, name=area)
    
        layout = go.Layout(
            scene_xaxis_visible=False, 
            scene_yaxis_visible=False,
            scene_zaxis_visible=False,
            showlegend=True
        )
    
        fig2 = go.Figure(
            data=[mesh3D], 
            layout=layout
        )
        fig_data = fig_data + fig2.data
        
    fig3 = go.Figure(data=fig_data, layout=fig1.layout)
    if not show_axis:
        fig3.update_layout(layout)
    
    return(fig3)

def get_name_to_color(positions, structure_level, color_list=None, unknown_color="#adadad"):
    positions_by_structure = positions.sort_values(structure_level)
    structure_names = positions_by_structure[structure_level].unique()
    name_to_color = {}
    if not color_list:
        color_list = [matplotlib.colors.to_hex(x) for x in plt.cm.tab10.colors]
    for i, pos in enumerate(structure_names):
        name_to_color[pos] = color_list[i % len(color_list)]
    # add a grey for unknown
    name_to_color["unknown"] = unknown_color
    return name_to_color


def generate_glass_brain(positions, structure_level, color_list=None, name_to_color=None, unknown_color="#adadad"):
    # hack to get things in the same order
    positions = positions.sort_values(structure_level)
    if not name_to_color:
        name_to_color = get_name_to_color(positions, structure_level, color_list, unknown_color)
    fig1 = px.scatter_3d(
        positions, x="x", y="y", z="z", 
        color=structure_level, 
        labels={structure_level: "Structure"},
        color_discrete_map=name_to_color)
    fig = plotly_add_glass_brain(None, fig1, "SA", areas='all', show_axis=False)
    fig.update_layout(
        autosize=True,
        width=750,
        height=700,
    )
    camera = dict(
        eye=dict(x=1.3, y=1, z=0.1)
    )
    temp_grid = dict(
        showgrid=False, 
        zeroline=False, 
        showticklabels=False
    )
    fig.update_layout(
        autosize=False,
        width=1000,
        height=700,
        xaxis=temp_grid,
        yaxis=temp_grid,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend={'traceorder':'normal'},
        legend_title_text=None,
        scene_camera=camera
    )
    return fig

def visualize_weights(positions, weights, structure_level, interval_size=100, color_list=None, name_to_color=None, cmap=None, unknown_color="#adadad", add_region=True, mean_weights_df=None, ax=None, fig=None):
    # get re-ordered indexes
    pos_unit_sorted = positions.sort_values(by="PseudoUnitID")
    pos_unit_sorted["np_idx"] = np.arange(0, len(pos_unit_sorted))
    pos_unit_sorted = pos_unit_sorted[pos_unit_sorted[structure_level] != "unknown"]
    if mean_weights_df is None:
        pos_structure_sorted = pos_unit_sorted.sort_values(by=structure_level)
        reordered_idxs = pos_structure_sorted.np_idx.values
    else: 
        merged = pd.merge(pos_unit_sorted, mean_weights_df, on="np_idx")
        pos_structure_sorted = merged.sort_values(by=[structure_level, "weight"], ascending=[True, False])
        # pos_structure_sorted = pos_structure_sorted.groupby(structure_level).apply(lambda x: x.iloc[:int(len(x) * 0.5)]).reset_index(drop=True)
        reordered_idxs = pos_structure_sorted.np_idx.values

    # get lengths of vertical bands on the left
    lens = pos_structure_sorted.groupby(structure_level).apply(lambda x: len(x)).values
    lens_accum = list(accumulate(lens))
    reg_start = lens_accum[:-1].insert(0, 0)
    reg_end = list(np.array(lens_accum[1:]) - 1)
    lines = np.array(lens_accum[0:-1]) - 0.5

    if not name_to_color:
        name_to_color = get_name_to_color(positions, structure_level, color_list, unknown_color)

    # reorder by temp then ant
    reordered = weights[reordered_idxs, :]

    # sort structure names
    positions_by_structure = positions.sort_values(structure_level)
    structure_names = positions_by_structure[structure_level].unique()

    if not ax: 
        _, ax = plt.subplots(figsize=(8, 15))
    if cmap is None:
        colors = ax.matshow(reordered, aspect='auto')
    else: 
        colors = ax.matshow(reordered, aspect='auto', cmap=cmap)
    # # tick_labels = np.array([-1.2, -0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9, 1.2])
    # ratio = 1000 / interval_size
    # tick_labels = np.array([-1, -0.5, 0, 0.5, 1.0, 1.5])
    # tick_pos = (tick_labels + 1.3) * ratio - 0.5
    # # fig.colorbar(colors)
    # # axis = np.arange(0, 28, 3)s
    # # labels = np.around((axis - 13) * 0.1, 1)
    # ax.set_xticks(tick_pos)
    # ax.set_xticklabels(tick_labels)
    # ax.xaxis.tick_bottom()
    # ax.set_xlabel("Time Relative to Feedback (s)")
    ax.get_yaxis().set_visible(False)
    ax.set_ylabel([])
    # y_axis = np.arange(0, 59, 5)
    # ax.set_yticks(y_axis)
    # ax.set_yticklabels(y_axis)

    boundaries = np.insert(np.insert(lines, len(lines), reordered.shape[0] - 0.5), 0, -0.5)
    for line in lines:
        ax.axhline(line, color='white', linestyle="dotted", linewidth=1.5)

    for i in range(len(boundaries)-1):
        structure_name = structure_names[i]
        rect = patches.Rectangle(
            (
                -1.3 - 1,
                (boundaries[i])#+boundaries[i+1]) / 2
            ),
            1.2,
            (boundaries[i+1]-boundaries[i]),
            edgecolor=name_to_color[structure_name],
            facecolor=name_to_color[structure_name],
            clip_on=False
        )
        if add_region:
            ax.add_patch(rect)
    gray_rect = patches.Rectangle(
        (4.5, -1.5), 8, 0.5,
        edgecolor="gray",
        facecolor="gray",
        clip_on=False,
    )
    # ax.add_patch(gray_rect)
    # ax.axvline(13.48, color="gray", linestyle="dotted", linewidth=1.5)

def get_fr_np_array(fr_df, mode):
    return fr_df.sort_values(by=["TrialNumber", "TimeBins"])[mode].to_numpy().reshape(
        len(fr_df.TrialNumber.unique()), 
        len(fr_df.TimeBins.unique())
    )

def plot_mean_frs_by_group(sess_name, unit, frs, beh, group_name, pos, ax, mode="FiringRate", group_colors=None, group_label=None, group_order=None, set_ax=True):
    groups = group_order if group_order else beh[group_name].unique()
    for group in groups:
        trials = beh[beh[group_name] == group].TrialNumber
        group_frs = frs[(frs.TrialNumber.isin(trials)) & (frs.UnitID == unit)]
        vals = get_fr_np_array(group_frs, mode)
        color = group_colors[group] if group_colors else None
        label = group_label[group] if group_label else group
        visualize_accuracy_across_time_bins(
            vals.T,
            1.3, 1.5, 0.1,
            # 0.5, 0.5, 0.05,
            ax,
            label=label,
            right_align=True, 
            sem=True,
            color=color
        )
    unit_pos = pos[pos.UnitID == unit].manual_structure.unique()[0]
    ax.legend()
    if set_ax:
        ax.set_title(f"Session {sess_name} Unit {unit} ({unit_pos})")
        ax.axvspan(-0.8, 0, alpha=0.3, color='gray')
        ax.axvline(0.098, alpha=0.3, color='gray', linestyle='dashed')
        ax.set_xlabel("Time Relative to Feedback (s)")
    # ax.set_ylabel("Firing Rate (Hz)")


def plot_mean_frs_by_group_stim_on(sess_name, unit, frs, beh, group_name, pos, ax, mode="FiringRate", group_colors=None):
    groups = beh[group_name].unique()
    for group in groups:
        trials = beh[beh[group_name] == group].TrialNumber
        group_frs = frs[(frs.TrialNumber.isin(trials)) & (frs.UnitID == unit)]
        vals = get_fr_np_array(group_frs, mode)
        color = group_colors[group] if group_colors else None
        visualize_accuracy_across_time_bins(
            vals.T,
            0.5, 0.5, 0.1,
            ax,
            label=group,
            right_align=True, 
            sem=True,
            color=color
        )
    unit_pos = pos[pos.UnitID == unit].manual_structure.unique()[0]
    ax.set_title(f"Session {sess_name} Unit {unit} ({unit_pos})")
    ax.legend()
    ax.set_xlabel("Time Relative to Stim Onset (s)")
    ax.set_ylabel("Firing Rate (Hz)")

def plot_mean_sterrs_by_bin(df, data_column, bin_column, ax, label, num_bins):
    """
    Plots data by relative block position
    """
    means = df.groupby(bin_column)[data_column].mean()
    stds = df.groupby(bin_column)[data_column].std()
    # bin_size = 1 / num_bins
    # time_bins = np.arange(0, 1, bin_size)
    mean_line, = ax.plot(means.index, means, linewidth=2)
    sterr = stds / np.sqrt(len(stds))
    std_line = ax.fill_between(means.index, means - sterr, means + sterr, alpha=0.5, label=label)

def plot_bars_by_cat(df, data_column, cat_column, ax, order=None):
    """
    Plots data by category
    """
    sns.barplot(data=df, x=cat_column, y=data_column, capsize=.1, errorbar='se', order=order, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)


def plot_and_calc_correlation(var_a, var_b, ax):
    """
    Plots a scatterplot of two variables, fits a line
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(var_a, var_b)
    ax.scatter(var_a, var_b, alpha=0.3, color="black")
    ax.plot(var_a, var_a * slope + intercept)
    return slope, intercept, r_value, p_value, std_err

def sns_plot_correlation(df, x_col, y_col, ax):
    """
    Same as above just taking a df and using seaborn scatter 
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(df[x_col], df[y_col])
    sns.scatterplot(df, x=x_col, y=y_col, alpha=0.3, color="black", ax=ax)
    ax.plot(df[x_col], df[x_col] * slope + intercept)
    return slope, intercept, r_value, p_value, std_err

def plot_accs_seaborn(datas, labels, pre_interval, interval_size, ax):
    dfs = []
    for i, data in enumerate(datas): 
        df = pd.DataFrame(data).reset_index(names=["Time"])
        df["Time"] = (df["Time"] * interval_size + interval_size - pre_interval) / 1000
        df = df.melt(id_vars="Time", value_vars=list(range(data.shape[1])), var_name="run", value_name="Accuracy")
        df["label"] = labels[i]
        dfs.append(df)
    res = pd.concat(dfs)
    sns.lineplot(res, x="Time", y="Accuracy", hue="label", linewidth=3, ax=ax)


def visualize_ccpg_value(args, df, ax, hue_col="condition"):
    sns.lineplot(df, x="Time", y="Accuracy", hue=hue_col, linewidth=3, errorbar="se", ax=ax)
    # # add estimated chance
    ax.axhline(1/2, color='black', linestyle='dotted', label="Estimated Chance")
    if args.trial_event == "FeedbackOnset":
        ax.axvspan(-0.8, 0, alpha=0.3, color='gray')
    ax.legend()
    ax.set_ylabel("Decoder Accuracy")
    ax.set_xlabel(f"Time Relative to {args.trial_event}")
    ax.set_title(f"Subject {args.subject} CCGP of value, {args.regions} regions")

def visualize_preferred_beliefs(args, df, ax, p_vals=None, hue_col="condition", palette=None, show_shuffles=True, ylims=[0.44, 1]):
    if not show_shuffles:
        df = df[~df.condition.str.contains("shuffle")]
    else: 
        df["mode"] = df["mode"].apply(lambda x: "shuffle" if "shuffle" in x else x)
    df["mode"] = df["mode"].map(MODE_TO_DISPLAY_NAMES)
    acc_line = sns.lineplot(df, x="Time", y="Accuracy", hue=hue_col, linewidth=3, ax=ax, palette=palette, errorbar="se")
    ax.axhline(y=0.5, linestyle="dotted", color="black")
    if p_vals is not None: 
        for thresh, lw in SIG_LEVELS:
            sigs = p_vals[p_vals["p"] < thresh]
            for _, row in sigs.iterrows():
                # color = acc_line.lines[0].get_color()
                ax.hlines(y=0.46, xmin=row.Time - 0.1, xmax=row.Time, linewidth=lw, color="black")
        ax.set_ylim(bottom=ax.get_ylim()[0] - 0.2)

    if ylims: 
        ax.set_ylim(ylims)
    ax.legend()
    ax.set_ylabel("Accuracy")
    ax.set_xlabel(f"Time Relative to {args.trial_event}")

def visualize_unit_raster(subject, session, pseudo_unit_id, beh, trial_interval, ax, hue_order=None, palette=None):
    """
    Visualizes a raster plot of unit activity given some behavior condition
    X axis will be time from even in seconds
    Y axis will be trials, sorted by condition
    colors will be by condition
    """
    spike_times = spike_general.get_spike_times(None, subject, session, species_dir="/data")
    intervals = behavioral_utils.get_trial_intervals(
        beh, 
        trial_interval.event, 
        trial_interval.pre_interval, 
        trial_interval.post_interval
    )
    spike_by_trial_interval = spike_utils.get_spikes_by_trial_interval(spike_times, intervals)
    spike_by_trial_interval.TrialNumber = spike_by_trial_interval.TrialNumber.astype(int)

    unit_spikes = spike_by_trial_interval[spike_by_trial_interval.UnitID == int(pseudo_unit_id % 100)].copy()
    unit_spikes["X"] = (unit_spikes.SpikeTimeFromStart - trial_interval.pre_interval) / 1000
    sorted_beh = beh.sort_values(by=["condition", "TrialNumber"], ascending=False)
    sorted_beh["Y"] = range(len(sorted_beh))
    unit_spikes = pd.merge(unit_spikes, sorted_beh[["TrialNumber", "Y", "condition"]], on="TrialNumber")
    sns.scatterplot(unit_spikes, x="X", y="Y", hue="condition", marker=".", edgecolor="none", s=20, ax=ax, hue_order=hue_order, palette=palette)
    # sns.scatterplot(unit_spikes, x="X", y="Y", hue="condition", marker="|", linewidths=1, s=100, ax=ax, hue_order=hue_order, palette=palette)


def visualize_cond_psth(beh, frs, unit_id, cond, ax, num_bins=None):
    """
    Plot a psth for neural activity for some unit grouped by condition. 
    If condition is continuous, uses num_bins to bin data first
    """
    frs = frs[frs.PseudoUnitID == unit_id]
    if num_bins is not None:
        out, bins = pd.cut(beh[cond], 10, labels=False, retbins=True)
        beh[f"{cond}Label"] = bins[out]
        cond = f"{cond}Label"
    sns.lineplot(pd.merge(beh, frs, on="TrialNumber"), x="Time", y="FiringRate", hue=cond, errorbar="se", ax=ax)

def visualize_cond_correlations(beh, frs, unit_id, cond, ax):
    """
    Visualizes scatter and regression line of some condition and firing rate
    Returns r, p value of regression
    """
    frs = frs[frs.PseudoUnitID == unit_id]
    mean_frs = frs.groupby("TrialNumber").FiringRate.mean().reset_index(name="Mean FR")
    merged = pd.merge(beh, mean_frs, on="TrialNumber")
    sns.scatterplot(merged, x=cond, y="Mean FR", ax=ax)
    slope, intercept, r_value, p_value, std_err = stats.linregress(merged[cond], merged["Mean FR"])
    ax.plot(merged[cond], merged[cond] * slope + intercept, color="black", linewidth=2)
    return r_value, p_value

def visualize_cross_time(args, cross_res, decoder_res, ax, cbar=True, vmin=None, vmax=None):
    shuffles = decoder_res[decoder_res["mode"] == f"{args.mode}_shuffle"]
    shuffle_means = shuffles.groupby(["Time"]).Accuracy.mean().reset_index(name="ShuffleAccuracy")
    cross_res = pd.merge(cross_res, shuffle_means, left_on="TestTime", right_on="Time")
    if vmin is None: 
        vmin = shuffle_means.ShuffleAccuracy.min()
    cross_res["Accuracy"] = cross_res.apply(lambda x: vmin if x.Accuracy < x.ShuffleAccuracy else x.Accuracy, axis=1)
    pivoted = cross_res.pivot(index="TrainTime", columns="TestTime", values="Accuracy")
    sns.heatmap(pivoted, ax=ax, vmin=vmin, vmax=vmax, cbar=cbar)

def visualize_cross_time_with_sig(args, cross_res, ax, vmin, vmax, p_vals=None, sig_thresh=None):
    """
    Plots nans as white, non-significant cells as black
    Adjusts colormap accordingly
    """
    pivoted = cross_res.pivot(index="TrainTime", columns="TestTime", values="Accuracy")
    if sig_thresh is not None:
        sig_mask = p_vals.pivot(index="TrainTime", columns="TestTime", values="p")
        sig_mask = sig_mask >= sig_thresh
        pivoted[sig_mask] = -1
    cmap = ListedColormap(['black'] + sns.color_palette("rocket", 256).as_hex())
    boundaries = [-1, vmin] + list(np.linspace(vmin, vmax, 256))
    norm = BoundaryNorm(boundaries, cmap.N)
    sns.heatmap(pivoted, ax=ax, cmap=cmap, norm=norm, cbar=False, vmin=vmin, vmax=vmax)


def plot_combined_accs(args, by_dim=False, modes=None, with_pvals=True):
    if not modes: 
        stim_args = copy.deepcopy(args)
        stim_args.trial_event = "StimOnset"
        stim_res = belief_partitions_io.read_results(stim_args, FEATURES)

        fb_args = copy.deepcopy(args)
        fb_args.trial_event = "FeedbackOnsetLong"
        fb_res = belief_partitions_io.read_results(fb_args, FEATURES)
        if with_pvals:
            stim_pvals = pd.read_pickle(os.path.join(belief_partitions_io.get_dir_name(stim_args), f"{args.mode}_pvals.pickle"))
            fb_pvals = pd.read_pickle(os.path.join(belief_partitions_io.get_dir_name(fb_args), f"{args.mode}_pvals.pickle"))
        else: 
            stim_pvals, fb_pvals = None, None
    else: 
        stim_res = []
        fb_res = []
        for mode in modes: 
            stim_args = copy.deepcopy(args)
            stim_args.mode = mode
            stim_args.trial_event = "StimOnset"
            stim_res.append(belief_partitions_io.read_results(stim_args, FEATURES))

            fb_args = copy.deepcopy(args)
            fb_args.mode = mode
            fb_args.trial_event = "FeedbackOnsetLong"
            fb_res.append(belief_partitions_io.read_results(fb_args, FEATURES))
        stim_res = pd.concat(stim_res)
        fb_res = pd.concat(fb_res)
    if by_dim: 
        fb_res["mode"] = fb_res.apply(lambda x: "shuffle" if "shuffle" in x["mode"] else FEATURE_TO_DIM[x.feat], axis=1)
        stim_res["mode"] = stim_res.apply(lambda x: "shuffle" if "shuffle" in x["mode"] else FEATURE_TO_DIM[x.feat], axis=1)
    # fb_res = fb_res[fb_res.feat != "GREEN"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 9/3), sharey='row', width_ratios=[stim_res.Time.nunique(), fb_res.Time.nunique()])
    visualize_preferred_beliefs(stim_args, stim_res, ax1, p_vals=stim_pvals, hue_col="mode", palette=MODE_TO_COLOR)
    ax1.axvline(-.5, color='grey', linestyle='dotted', linewidth=3)
    ax1.axvline(0, color='grey', linestyle='dotted', linewidth=3)

    ax1.set_xlabel(f"Time to cards appear (s)")
    stim_ticks = [-1, -.5, 0, .5, 1]
    ax1.set_xticks(stim_ticks)
    ax1.set_xticklabels(stim_ticks)

    visualize_preferred_beliefs(fb_args, fb_res, ax2, p_vals=fb_pvals, hue_col="mode", palette=MODE_TO_COLOR)
    ax2.axvline(-.8, color='grey', linestyle='dotted', linewidth=3)
    ax2.axvline(0, color='grey', linestyle='dotted', linewidth=3)
    ax2.set_xlabel(f"Time to feedback (s)")
    fb_ticks = [-1.5, -1, -.5, 0, .5, 1, 1.5]
    ax2.set_xticks(fb_ticks)
    ax2.set_xticklabels(fb_ticks)
    format_plot([ax1, ax2])
    ax2.get_legend().remove()
    for line in ax1.legend().get_lines():
        line.set_linewidth(6)

    fig.tight_layout()
    return fig, (ax1, ax2)

def plot_combined_accs_by_attr(args, attr, values, num_shuffles=0, hue_order=None, palette=None, display_names=None):
    all_stim_res = []
    all_fb_res = []
    for val in values:
        print(val)
        setattr(args, attr, val)
        stim_args = copy.deepcopy(args)
        stim_args.trial_event = "StimOnset"
        stim_res = belief_partitions_io.read_results(stim_args, FEATURES, num_shuffles=num_shuffles)
        stim_res[attr] = val
        if num_shuffles > 0: 
            stim_res[attr] = stim_res.apply(lambda x: "shuffle" if "shuffle" in x["mode"] else x[attr], axis=1)
        all_stim_res.append(stim_res)

        fb_args = copy.deepcopy(args)
        fb_args.trial_event = "FeedbackOnsetLong"
        fb_res = belief_partitions_io.read_results(fb_args, FEATURES, num_shuffles=num_shuffles)
        fb_res[attr] = val
        if num_shuffles > 0: 
            fb_res[attr] = fb_res.apply(lambda x: "shuffle" if "shuffle" in x["mode"] else x[attr], axis=1)
        all_fb_res.append(fb_res)
    all_stim_res = pd.concat(all_stim_res)
    all_fb_res = pd.concat(all_fb_res)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3), sharey='row', width_ratios=[all_stim_res.Time.nunique(), all_fb_res.Time.nunique()])
    if display_names:
        all_stim_res[attr] = all_stim_res[attr].map(display_names)

    sns.lineplot(
        all_stim_res, x="Time", y="Accuracy", 
        hue=attr, hue_order=hue_order, 
        linewidth=3, ax=ax1, 
        palette=palette, errorbar="se"
    )


    ax1.axvline(-.5, color='grey', linestyle='dotted', linewidth=3)
    ax1.axvline(0, color='grey', linestyle='dotted', linewidth=3)

    ax1.set_xlabel(f"Time to cards appear (s)")
    stim_ticks = [-1, -.5, 0, .5, 1]
    ax1.set_xticks(stim_ticks)
    ax1.set_xticklabels(stim_ticks)
    if display_names:
        all_fb_res[attr] = all_fb_res[attr].map(display_names)
    sns.lineplot(
        all_fb_res, x="Time", y="Accuracy", 
        hue=attr, hue_order=hue_order, 
        linewidth=3, ax=ax2, 
        palette=palette, errorbar="se"
    )
    ax2.axvline(-.8, color='grey', linestyle='dotted', linewidth=3)
    ax2.axvline(0, color='grey', linestyle='dotted', linewidth=3)
    ax2.set_xlabel(f"Time to feedback (s)")
    fb_ticks = [-1.5, -1, -.5, 0, .5, 1, 1.5]
    ax2.set_xticks(fb_ticks)
    ax2.set_xticklabels(fb_ticks)
    format_plot([ax1, ax2])
    ax2.get_legend().remove()
    for line in ax1.legend().get_lines():
        line.set_linewidth(6)

    fig.tight_layout()
    return fig, (ax1, ax2)


def plot_combined_cross_accs(args, ignore_overlap=False):
    args.trial_event = "StimOnset"
    stim_res = belief_partitions_io.read_results(args, FEATURES)
    cross_stim_res = belief_partitions_io.read_cross_time_results(args, FEATURES, avg=True)
    args.model_trial_event = "FeedbackOnsetLong"
    fb_model_cross_stim_res = belief_partitions_io.read_cross_time_results(args, FEATURES, avg=True)
    if ignore_overlap:
        fb_model_cross_stim_res.loc[
            (fb_model_cross_stim_res.TrainTime < -0.8) | 
            (fb_model_cross_stim_res.TestTime >=0), 
            "Accuracy"
        ] = 0

    args.model_trial_event = None


    args.trial_event = "FeedbackOnsetLong"
    fb_res = belief_partitions_io.read_results(args, FEATURES)
    cross_fb_res = belief_partitions_io.read_cross_time_results(args, FEATURES, avg=True)
    args.model_trial_event = "StimOnset"
    stim_model_cross_fb_res = belief_partitions_io.read_cross_time_results(args, FEATURES, avg=True)
    args.model_trial_event = None

    if ignore_overlap:
        fb_model_cross_stim_res.loc[
            (fb_model_cross_stim_res.TrainTime < -0.8) | 
            (fb_model_cross_stim_res.TestTime >=0), 
            "Accuracy"
        ] = np.nan
        stim_model_cross_fb_res.loc[
            (stim_model_cross_fb_res.TrainTime >= 0) | 
            (stim_model_cross_fb_res.TestTime < -0.8), 
            "Accuracy"
        ] = np.nan

    fig, axs = plt.subplots(
        2, 2, figsize=(11, 10),                            
        width_ratios=[cross_stim_res.TestTime.nunique(), cross_fb_res.TestTime.nunique()],
        height_ratios=[cross_stim_res.TestTime.nunique(), cross_fb_res.TestTime.nunique()],
        sharex="col",
        sharey="row",
    )
    all_res = pd.concat((cross_stim_res, stim_model_cross_fb_res, fb_model_cross_stim_res, cross_fb_res))
    all_max = all_res.groupby(["TestTime", "TrainTime", "TrainEvent", "TestEvent"]).Accuracy.mean().max()

    all_decoder_res = pd.concat((stim_res, fb_res))
    shuffles = all_decoder_res[all_decoder_res["mode"] == f"{args.mode}_shuffle"]
    shuffle_means = shuffles.groupby(["Time"]).Accuracy.mean().reset_index(name="ShuffleAccuracy")
    all_min = shuffle_means.ShuffleAccuracy.mean()

    visualize_cross_time_with_sig(args, cross_stim_res, axs[0, 0], vmin=all_min, vmax=all_max)
    visualize_cross_time_with_sig(args, stim_model_cross_fb_res, axs[0, 1], vmin=all_min, vmax=all_max)
    visualize_cross_time_with_sig(args, fb_model_cross_stim_res, axs[1, 0], vmin=all_min, vmax=all_max)
    visualize_cross_time_with_sig(args, cross_fb_res, axs[1, 1], vmin=all_min, vmax=all_max)


    stim_ticks = [-.5, 0, .5, 1]
    stim_tick_pos = [st * 10 + 10 - 0.5 for st in stim_ticks]
    fb_ticks = [-1.5, -1, -.5, 0, .5, 1, 1.5]
    fb_tick_pos = [ft * 10 + 18 - 0.5 for ft in fb_ticks]

    cross_fix_pos, stim_on_pos = 4.5, 9.5
    card_fix_pos, fb_pos = 9.5, 17.5
    axs[0, 0].set_ylabel("Time to cards appear (s)")
    axs[0, 0].set_xlabel("")
    axs[0, 0].set_yticks(stim_tick_pos)
    axs[0, 0].set_yticklabels(stim_ticks)
    axs[0, 0].axvline(cross_fix_pos, color='white', linestyle='dotted', linewidth=3)
    axs[0, 0].axvline(stim_on_pos, color='white', linestyle='dotted', linewidth=3)
    axs[0, 0].axhline(cross_fix_pos, color='white', linestyle='dotted', linewidth=3)
    axs[0, 0].axhline(stim_on_pos, color='white', linestyle='dotted', linewidth=3)

    axs[0, 1].set_xlabel("")
    axs[0, 1].set_ylabel("")
    axs[0, 1].axhline(cross_fix_pos, color='white', linestyle='dotted', linewidth=3)
    axs[0, 1].axhline(stim_on_pos, color='white', linestyle='dotted', linewidth=3)
    axs[0, 1].axvline(card_fix_pos, color='white', linestyle='dotted', linewidth=3)
    axs[0, 1].axvline(fb_pos, color='white', linestyle='dotted', linewidth=3)

    axs[1, 0].set_xlabel("Time to cards appear (s)")
    axs[1, 0].set_ylabel("Time to feedback (s)")
    axs[1, 0].set_xticks(stim_tick_pos)
    axs[1, 0].set_xticklabels(stim_ticks)
    axs[1, 0].set_yticks(fb_tick_pos)
    axs[1, 0].set_yticklabels(fb_ticks)
    axs[1, 0].axvline(cross_fix_pos, color='white', linestyle='dotted', linewidth=3)
    axs[1, 0].axvline(stim_on_pos, color='white', linestyle='dotted', linewidth=3)
    axs[1, 0].axhline(card_fix_pos, color='white', linestyle='dotted', linewidth=3)
    axs[1, 0].axhline(fb_pos, color='white', linestyle='dotted', linewidth=3)

    axs[1, 1].set_xlabel("Time to feedback (s)")
    axs[1, 1].set_ylabel("")
    axs[1, 1].set_xticks(fb_tick_pos)
    axs[1, 1].set_xticklabels(fb_ticks)
    axs[1, 1].axvline(card_fix_pos, color='white', linestyle='dotted', linewidth=3)
    axs[1, 1].axvline(fb_pos, color='white', linestyle='dotted', linewidth=3)
    axs[1, 1].axhline(card_fix_pos, color='white', linestyle='dotted', linewidth=3)
    axs[1, 1].axhline(fb_pos, color='white', linestyle='dotted', linewidth=3)
    # axs[1, 1].set_yticks([])

    fig.supylabel("Train time")
    fig.supxlabel("Test time")
    fig.tight_layout()

    # Adjust subplots to make space for colorbar
    fig.subplots_adjust(right=0.85)
    # Create a single colorbar to the right of ax2
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(axs[1, 1].collections[0], cax=cbar_ax, orientation='vertical')
    cbar.set_label('Accuracy')

    # fig.tight_layout()
    return fig, axs


def plot_combined_cross_accs_trunc(args, alpha=0.01):
    args.trial_event = "StimOnset"
    stim_res = belief_partitions_io.read_results(args, FEATURES)
    cross_stim_res = belief_partitions_io.read_cross_time_results(args, FEATURES, avg=True)
    args.model_trial_event = "FeedbackOnsetLong"
    fb_stim_res = belief_partitions_io.read_cross_time_results(args, FEATURES, avg=True)
    # filter for training on fb period, testing before cards appear
    fb_stim_res = fb_stim_res[(fb_stim_res.TrainTime >= -0.8) & (fb_stim_res.TestTime < 0)]
    args.model_trial_event = None


    args.trial_event = "FeedbackOnsetLong"
    fb_res = belief_partitions_io.read_results(args, FEATURES)
    cross_fb_res = belief_partitions_io.read_cross_time_results(args, FEATURES, avg=True)
    args.model_trial_event = "StimOnset"
    stim_fb_res = belief_partitions_io.read_cross_time_results(args, FEATURES, avg=True)
    print(stim_fb_res.TrainTime.unique())
    print(stim_fb_res.TestTime.unique())
    # filter for training before cards appear, testing after fb
    stim_fb_res = stim_fb_res[(stim_fb_res.TrainTime < 0) & (stim_fb_res.TestTime >= -0.8)]
    stim_fb_res.loc[(stim_fb_res.TrainTime < 0) & (stim_fb_res.TestTime >= -0.8), "Accuracy"]
    print(len(stim_fb_res))
    args.model_trial_event = None

    layout = [
        ["ss", "ss", "n0", "sf"],
        ["ss", "ss", "n1", "n1"],
        ["n2", "n3", "ff", "ff"],
        ["fs", "n3", "ff", "ff"],
    ]
    # mosaic + some buffer for colorbar
    fig, axs = plt.subplot_mosaic(
        layout,
        figsize=(10, 10),
        width_ratios=(10, 8, 8, 23),
        height_ratios=(10, 8, 8, 23),
        constrained_layout=True
    )
    for i in range(4):
        axs[f"n{i}"].axis("off")
    # axs['fs'].sharex(axs['ss']) 
    # axs['sf'].sharex(axs['ff']) 
    # axs['fs'].sharey(axs['ff']) 
    # axs['sf'].sharey(axs['ss']) 

    all_res = pd.concat((cross_stim_res, stim_fb_res, fb_stim_res, cross_fb_res))
    all_max = all_res.groupby(["TestTime", "TrainTime", "TrainEvent", "TestEvent"]).Accuracy.mean().max()

    all_decoder_res = pd.concat((stim_res, fb_res))
    shuffles = all_decoder_res[all_decoder_res["mode"] == f"{args.mode}_shuffle"]
    shuffle_means = shuffles.groupby(["Time"]).Accuracy.mean().reset_index(name="ShuffleAccuracy")
    all_min = shuffle_means.ShuffleAccuracy.mean()

    visualize_cross_time_with_sig(args, cross_stim_res, axs["ss"], vmin=all_min, vmax=all_max)
    visualize_cross_time_with_sig(args, stim_fb_res, axs["sf"], vmin=all_min, vmax=all_max)
    visualize_cross_time_with_sig(args, fb_stim_res, axs["fs"], vmin=all_min, vmax=all_max)
    visualize_cross_time_with_sig(args, cross_fb_res, axs["ff"], vmin=all_min, vmax=all_max)


    stim_ticks = [-.5, 0, .5, 1]
    stim_tick_pos = [st * 10 + 10 - 0.5 for st in stim_ticks]
    fb_ticks = [-1.5, -1, -.5, 0, .5, 1, 1.5]
    fb_tick_pos = [ft * 10 + 18 - 0.5 for ft in fb_ticks]

    cross_fix_pos, stim_on_pos = 4.5, 9.5
    card_fix_pos, fb_pos = 9.5, 17.5
    axs["ss"].set_ylabel("Time to cards appear (s)")
    axs["ss"].set_xlabel("")
    axs["ss"].set_yticks(stim_tick_pos)
    axs["ss"].set_yticklabels(stim_ticks)
    axs["ss"].axvline(cross_fix_pos, color='white', linestyle='dotted', linewidth=3)
    axs["ss"].axvline(stim_on_pos, color='white', linestyle='dotted', linewidth=3)
    axs["ss"].axhline(cross_fix_pos, color='white', linestyle='dotted', linewidth=3)
    axs["ss"].axhline(stim_on_pos, color='white', linestyle='dotted', linewidth=3)
    axs["ss"].set_xticks([])


    axs["sf"].set_xlabel("")
    axs["sf"].set_ylabel("")
    axs["sf"].set_xticks([])
    axs["sf"].set_yticks([])

    # axs["sf"].axhline(cross_fix_pos, color='white', linestyle='dotted', linewidth=3)
    # axs["sf"].axhline(stim_on_pos, color='white', linestyle='dotted', linewidth=3)
    # axs["sf"].axvline(card_fix_pos, color='white', linestyle='dotted', linewidth=3)
    # axs["sf"].axvline(fb_pos, color='white', linestyle='dotted', linewidth=3)

    # axs["fs"].set_xlabel("Time to cards appear (s)")
    # axs["fs"].set_ylabel("Time to feedback (s)")
    # axs["fs"].set_xticks(stim_tick_pos)
    # axs["fs"].set_xticklabels(stim_ticks)
    # axs["fs"].set_yticks(fb_tick_pos)
    # axs["fs"].set_yticklabels(fb_ticks)
    # axs["fs"].axvline(cross_fix_pos, color='white', linestyle='dotted', linewidth=3)
    # axs["fs"].axvline(stim_on_pos, color='white', linestyle='dotted', linewidth=3)
    # axs["fs"].axhline(card_fix_pos, color='white', linestyle='dotted', linewidth=3)
    # axs["fs"].axhline(fb_pos, color='white', linestyle='dotted', linewidth=3)

    axs["ff"].set_xlabel("Time to feedback (s)")
    axs["ff"].set_ylabel("")
    axs["ff"].set_xticks(fb_tick_pos)
    axs["ff"].set_xticklabels(fb_ticks)
    axs["ff"].axvline(card_fix_pos, color='white', linestyle='dotted', linewidth=3)
    axs["ff"].axvline(fb_pos, color='white', linestyle='dotted', linewidth=3)
    axs["ff"].axhline(card_fix_pos, color='white', linestyle='dotted', linewidth=3)
    axs["ff"].axhline(fb_pos, color='white', linestyle='dotted', linewidth=3)
    axs["ff"].set_yticks([])

    # fig.supylabel("Train time")
    # fig.supxlabel("Test time")

    # # Adjust subplots to make space for colorbar
    # fig.subplots_adjust(right=0.85)
    # # Create a single colorbar to the right of ax2
    # cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    # fig.colorbar(axs["ff"].collections[0], ax=axs["ff"], orientation='vertical', location='right', label='Accuracy')

    # fig.tight_layout()
    return fig, axs


def plot_combined_cross_accs_no_diag(args, alpha=0.01):
    stim_args = copy.deepcopy(args)
    stim_args.trial_event = "StimOnset"
    stim_res = belief_partitions_io.read_results(stim_args, FEATURES)
    cross_stim_res = belief_partitions_io.read_cross_time_results(stim_args, FEATURES, avg=True)

    fb_args = copy.deepcopy(args)
    fb_args.trial_event = "FeedbackOnsetLong"
    fb_res = belief_partitions_io.read_results(fb_args, FEATURES)
    cross_fb_res = belief_partitions_io.read_cross_time_results(fb_args, FEATURES, avg=True)

    stim_pvals = pd.read_pickle(os.path.join(belief_partitions_io.get_dir_name(stim_args), f"{args.mode}_cross_p_vals.pickle"))
    fb_pvals = pd.read_pickle(os.path.join(belief_partitions_io.get_dir_name(fb_args), f"{args.mode}_cross_p_vals.pickle"))

    stim_n = cross_stim_res.TestTime.nunique()
    fb_n = cross_fb_res.TestTime.nunique()
    ratio = 5

    # Define layout using a nested list (None means empty space)
    layout = [
        ["stim", "fb"],
        ["empty", "fb"],  # A spans two rows (taller), B spans one (shorter)
    ]
    # mosaic + some buffer for colorbar
    fig, axs = plt.subplot_mosaic(
        layout,
        figsize=((stim_n + fb_n + 5) / ratio, fb_n / ratio),
        width_ratios=(stim_n, fb_n),
        height_ratios=(stim_n, fb_n - stim_n),
        constrained_layout=True
    )
    stim_ax = axs["stim"]
    fb_ax = axs["fb"]
    axs["empty"].axis("off")

    # fig = plt.figure(figsize=((stim_n + fb_n) / ratio, fb_n / ratio), constrained_layout=True)
    # # in rows x cols
    # gs = gridspec.GridSpec(fb_n, stim_n + fb_n, figure=fig)
    
    # stim_ax = fig.add_subplot(gs[:stim_n, :stim_n])
    # fb_ax = fig.add_subplot(gs[:fb_n, stim_n:])

    all_res = pd.concat((cross_stim_res, cross_fb_res))
    all_max = all_res.groupby(["TestTime", "TrainTime", "TrainEvent", "TestEvent"]).Accuracy.mean().max()

    all_decoder_res = pd.concat((stim_res, fb_res))
    shuffles = all_decoder_res[all_decoder_res["mode"] == f"{args.mode}_shuffle"]
    shuffle_means = shuffles.groupby(["Time"]).Accuracy.mean().reset_index(name="ShuffleAccuracy")
    all_min = shuffle_means.ShuffleAccuracy.mean()

    cmap = plt.get_cmap('rocket').copy()
    cmap.set_bad(color='black')  # grey for NaNs
    visualize_cross_time_with_sig(args, cross_stim_res, stim_pvals, stim_ax, thresh=alpha, cmap=cmap, vmin=all_min, vmax=all_max)
    visualize_cross_time_with_sig(args, cross_fb_res, fb_pvals, fb_ax, thresh=alpha, cmap=cmap, vmin=all_min, vmax=all_max)


    stim_ticks = [-.5, 0, .5, 1]
    stim_tick_pos = [st * 10 + 10 - 0.5 for st in stim_ticks]
    fb_ticks = [-1.5, -1, -.5, 0, .5, 1, 1.5]
    fb_tick_pos = [ft * 10 + 18 - 0.5 for ft in fb_ticks]
    cross_fix_pos, stim_on_pos = 4.5, 9.5
    card_fix_pos, fb_pos = 9.5, 17.5

    stim_ax.set_ylabel("Time to cards appear (s)")
    stim_ax.set_xlabel("Time to cards appear (s)")
    stim_ax.set_xticks(stim_tick_pos)
    stim_ax.set_xticklabels(stim_ticks)
    stim_ax.set_yticks(stim_tick_pos)
    stim_ax.set_yticklabels(stim_ticks)
    stim_ax.axvline(cross_fix_pos, color='white', linestyle='dotted', linewidth=3)
    stim_ax.axvline(stim_on_pos, color='white', linestyle='dotted', linewidth=3)
    stim_ax.axhline(cross_fix_pos, color='white', linestyle='dotted', linewidth=3)
    stim_ax.axhline(stim_on_pos, color='white', linestyle='dotted', linewidth=3)

    fb_ax.set_xlabel("Time to feedback (s)")
    fb_ax.set_ylabel("Time to feedback (s)")
    fb_ax.set_xticks(fb_tick_pos)
    fb_ax.set_xticklabels(fb_ticks)
    fb_ax.set_yticks(fb_tick_pos)
    fb_ax.set_yticklabels(fb_ticks)
    fb_ax.axvline(card_fix_pos, color='white', linestyle='dotted', linewidth=3)
    fb_ax.axvline(fb_pos, color='white', linestyle='dotted', linewidth=3)
    fb_ax.axhline(card_fix_pos, color='white', linestyle='dotted', linewidth=3)
    fb_ax.axhline(fb_pos, color='white', linestyle='dotted', linewidth=3)

    fig.supylabel("Train time")
    fig.supxlabel("Test time")

    # # Adjust subplots to make space for colorbar
    # fig.subplots_adjust(right=0.85)
    # # Create a single colorbar to the right of ax2
    # cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    # cbar = fig.colorbar(fb_ax.collections[0], cax=cbar_ax, orientation='vertical')
    # cbar.set_label('Accuracy')
    fig.colorbar(fb_ax.collections[0], ax=fb_ax, orientation='vertical', location='right', label='Accuracy')


    # fig.tight_layout()
    return fig, (stim_ax, fb_ax)



def plot_pop_heatmap_by_time(all_data, value_col, time_col="Time", region_level="whole_pop", orders=None, event_labels={}):
    """
    """

    unit_counts = all_data.groupby(region_level).PseudoUnitID.nunique().values
    regions_to_idx = {r: i for i, r in enumerate(np.sort(all_data[region_level].unique()))}
    trial_events = all_data.trial_event.unique()
    width_ratios = [all_data[all_data.trial_event == e][time_col].nunique() for e in trial_events]

    fig, axs = plt.subplots(
        len(unit_counts), len(trial_events), 
        figsize=(12, 17), 
        width_ratios=width_ratios,
        height_ratios=unit_counts, 
        sharex="col",
        sharey="row",
        squeeze=False
    )

    vmin = all_data[value_col].min()
    vmax = all_data[value_col].max()

    def plot_region(reg_conts):        
        region = reg_conts.name

        for i, event in enumerate(trial_events):
            event_conts = reg_conts[reg_conts.trial_event == event]
            # print(reg_conts.co)
            pivoted = event_conts.pivot(index="PseudoUnitID", columns=time_col, values=value_col)
            # print(pivoted.index)
            if orders: 
                order = orders[region]
                pivoted = pivoted.loc[order]
            ax = axs[regions_to_idx[region], i]
            sns.heatmap(pivoted, vmin=vmin, vmax=vmax, cbar=False, ax=ax, cmap="viridis")
            ax.set_yticks([])
            ax.set_yticklabels("")
            ax.set_xlabel("")
            ax.set_ylabel("")
        cross_fix_pos, stim_on_pos = 4.5, 9.5
        card_fix_pos, fb_pos = 9.5, 17.5
        axs[regions_to_idx[region], 0].set_title(region)
        axs[regions_to_idx[region], 0].axvline(cross_fix_pos, color='grey', linestyle='dotted', linewidth=3)
        axs[regions_to_idx[region], 0].axvline(stim_on_pos, color='grey', linestyle='dotted', linewidth=3)
        axs[regions_to_idx[region], 1].axvline(card_fix_pos, color='grey', linestyle='dotted', linewidth=3)
        axs[regions_to_idx[region], 1].axvline(fb_pos, color='grey', linestyle='dotted', linewidth=3)
    all_data.groupby(region_level).apply(plot_region)

    # for i, event in enumerate(trial_events):
    #     axs[-1, i].set_xlabel(event_labels.get(event, f"Time to {event} (s)"))
    stim_ticks = [-.5, 0, .5, 1]
    stim_tick_pos = [st * 10 + 10 - 0.5 for st in stim_ticks]
    fb_ticks = [-1.5, -1, -.5, 0, .5, 1, 1.5]
    fb_tick_pos = [ft * 10 + 18 - 0.5 for ft in fb_ticks]
    
    axs[-1, 0].set_xlabel("Time to card appear (s)")
    axs[-1, 0].set_xticks(stim_tick_pos)
    axs[-1, 0].set_xticklabels(stim_ticks)

    fb_ticks = [-1.5, -1, -.5, 0, .5, 1, 1.5]
    axs[-1, 1].set_xlabel("Time to feedback (s)")
    axs[-1, 1].set_xticks(fb_tick_pos)
    axs[-1, 1].set_xticklabels(fb_ticks)



    fig.tight_layout()
    # Adjust subplots to make space for colorbar
    fig.subplots_adjust(right=0.85)

    # # Create a single colorbar to the right of ax2
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(axs[-1, -1].collections[0], cax=cbar_ax, orientation='vertical')
    cbar.set_label('Contribution')
    return fig, axs

def plot_belief_partition_psth(unit_id, feat, args):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    args.feat = feat
    session = int(unit_id / 100)
    beh, frs = load_data(session, args, return_merged=False)
    frs = frs[frs.PseudoUnitID == unit_id]

    order = ["Low", "High"]
    colors = ["tab:green", "tab:purple"]
    sns.lineplot(pd.merge(frs, beh, on="TrialNumber"), x="Time", y="FiringRate", errorbar="se", hue="BeliefConf", hue_order=order, palette=colors, ax=ax1)
    ax1.set_xlabel(f"Time to {args.trial_event} (s)")
    ax1.set_title("Confidence")

    sub_beh = beh[beh.BeliefPartition.isin([f"High {args.feat}", f"High Not {args.feat}"])]
    order = [f"High Not {args.feat}", f"High {args.feat}"]
    colors = ["tab:blue", "tab:red"]

    sns.lineplot(pd.merge(frs, sub_beh, on="TrialNumber"), x="Time", y="FiringRate", errorbar="se", hue="BeliefPartition", hue_order=order, palette=colors, ax=ax2)
    ax2.set_xlabel(f"Time to {args.trial_event} (s)")
    ax2.set_title("Preference")

    order = ["Low", f"High Not {args.feat}", f"High {args.feat}"]
    colors = ["tab:green", "tab:blue", "tab:red"]

    sns.lineplot(pd.merge(frs, beh, on="TrialNumber"), x="Time", y="FiringRate", errorbar="se", hue="BeliefPartition", hue_order=order, palette=colors, ax=ax3)
    ax3.set_xlabel(f"Time to {args.trial_event} (s)")
    ax3.set_title("Belief Partitions")
    fig.tight_layout()
    return fig, (ax1, ax2, ax3)


def plot_psth_both_events(mode, unit_id, feat, args, pval_res=None):
    fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharey='row', sharex='col', width_ratios=[20, 33], height_ratios=[1, 2])
    args.feat = feat
    session = int(unit_id / 100)
    order = (MODE_TO_DIRECTION_LABELS[mode]["high"], MODE_TO_DIRECTION_LABELS[mode]["low"])
    colors = [MODE_TO_COLOR[MODE_TO_DISPLAY_NAMES[mode]], "black"]

    for i, trial_event in enumerate(["StimOnset", "FeedbackOnsetLong"]):
        args.trial_event = trial_event
        trial_interval = get_trial_interval(trial_event)
        args.trial_interval = trial_interval
        beh, frs = load_data(session, args, return_merged=False, use_x=True)
        beh = behavioral_utils.get_label_by_mode(beh, mode)
        frs = frs[frs.PseudoUnitID == unit_id]
        sns.lineplot(
            pd.merge(frs, beh, on="TrialNumber"), 
            x="Time", y="FiringRate", 
            linewidth=3,
            errorbar="se", 
            hue="condition", hue_order=order, 
            palette=colors, 
            ax=axs[0, i]
        )
        subject = behavioral_utils.get_sub_for_session(session)
        sub_beh = beh.sample(500) if len(beh) > 500 else beh
        visualize_unit_raster(subject, session, unit_id, sub_beh, trial_interval, ax=axs[1, i], hue_order=order, palette=colors)

    #add significance bars after both event have already been plotted
    if pval_res is not None:
        _, ymax_left = axs[0, i].get_ylim()
        _, ymax_right = axs[0, i].get_ylim()
        ybar = max((ymax_left, ymax_right)) * 1.05
        for i, trial_event in enumerate(["StimOnset", "FeedbackOnsetLong"]):
            res = pval_res[(pval_res.trial_event == trial_event) & (pval_res.PseudoUnitID == unit_id)]
            for thresh_str, lw in ANOVA_SIG_LEVELS:
                sigs = res[res["p"] == thresh_str]
                for _, row in sigs.iterrows():
                    # color = acc_line.lines[0].get_color()
                    axs[0, i].plot([row.Time - 0.05, row.Time + 0.05], [ybar, ybar], linewidth=lw, color='black')
                    # axs[0, i].hlines(y=-2, xmin=row.Time - 0.25, xmax=row.Time + 0.25, linewidth=lw, color="black")

    for row_idx in range(2):
        (ax1, ax2) = axs[row_idx, :]
        ax1.axvline(-.5, color='grey', linestyle='dotted', linewidth=3)
        ax1.axvline(0, color='grey', linestyle='dotted', linewidth=3)
        ax2.axvline(-.8, color='grey', linestyle='dotted', linewidth=3)
        ax2.axvline(0, color='grey', linestyle='dotted', linewidth=3)
    
    axs[0, 0].set_ylabel("Firing Rate (Hz)")

    ax1, ax2 = axs[1, :]
    ax1.set_ylabel("Trials")
    ax1.set_xlabel(f"Time to cards appear (s)")
    stim_ticks = [-1, -.5, 0, .5, 1]
    ax1.set_xticks(stim_ticks)
    ax1.set_xticklabels(stim_ticks)

    ax2.set_xlabel(f"Time to feedback (s)")
    fb_ticks = [-1.5, -1, -.5, 0, .5, 1, 1.5]
    ax2.set_xticks(fb_ticks)
    ax2.set_xticklabels(fb_ticks)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].get_legend().remove()
    axs[0, 1].get_legend().remove()
    axs[1, 1].get_legend().remove()
    axs[1, 0].get_legend().remove()

    if mode == "choice":
        labels = [f"Selected card w. {feat}", f"Selected card w. no {feat}"]
    elif mode == "pref":
        labels = [f"Prefers {feat}", "Prefers other"]
    legend = fig.legend(
        handles, labels,
        loc='upper left',
        bbox_to_anchor=(0.37, 0.7),
        frameon=True,
        borderaxespad=0.0
    )
    for line in legend.get_lines():
        line.set_linewidth(6)
    format_plot(axs)
    fig.tight_layout()
    # fig.subplots_adjust()
    return fig, axs

def plot_cosine_sim_between_conf_pref(args, include_shuffle=True, consider_norm=True):
    """
    Plots the distribution of cosine similarie
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey='row', width_ratios=[20, 33])
    for i, event in enumerate(["StimOnset", "FeedbackOnsetLong"]):
        args = copy.deepcopy(args)
        args.trial_event = event
        both_models = []
        for mode in ["pref", "conf"]:
            args.mode = mode
            models = belief_partitions_io.read_models(args, FEATURES)
            models["weights"] = models.apply(lambda x: x.models.coef_[0, :] - x.models.coef_[1, :], axis=1)
            if consider_norm:
                models["weights"] = models.apply(lambda x: x.weights / np.sqrt(x.models.model.norm.running_var.detach().cpu().numpy() + 1e-5), axis=1)

            both_models.append(models)
        sim_res = classifier_utils.get_cross_cond_cosine_sim_of_weights(both_models[0], both_models[1])
        sns.lineplot(sim_res, x="Time", y="cosine_sim", linewidth=3, color="black", ax=axs[i])
        axs[i].set_xlabel(f"Time to {event}")
    if include_shuffle:
        bounds = classifier_utils.get_shuffled_cosine_sim_of_weights(both_models[0])
        for event_idx in range(2):
            axs[event_idx].axhspan(bounds[0], bounds[1], alpha=0.3, color='gray')


    fig.tight_layout()

def format_plot(
    axs,
    linewidth=1,
    ticklength=8,
    ticklabelsize=12,
    axislabelsize=13,
    tickwidth=1,
    rightspine=False,
    leftspine=True,
    topspine=False,
    bottomspine=True,
    ):
    if type(axs) is not list and type(axs) is not np.array and type(axs) is not np.ndarray:
        axs = [axs]
    if type(axs) is np.ndarray:
        axs = axs.flatten()
    for ax in axs:
        ax.spines['top'].set_visible(topspine)
        ax.spines['right'].set_visible(rightspine)
        ax.spines['bottom'].set_visible(bottomspine)
        ax.spines['left'].set_visible(leftspine)
        ax.spines['bottom'].set_linewidth(linewidth)
        ax.spines['left'].set_linewidth(linewidth)
        ax.tick_params(axis='both', length=ticklength, labelsize=ticklabelsize, width=tickwidth)
        ax.xaxis.label.set_size(axislabelsize)
        ax.yaxis.label.set_size(axislabelsize)


def compute_bar_positions_from_df(df, x, y, hue=None, group_width=0.8):
    """
    Compute expected bar center positions and heights (mean + se) from a dataframe.

    Returns:
        dict: {(category, hue): (x_position, bar_height)}
    """
    bar_positions = {}

    if hue:
        cond_order = df[x].unique()
        hue_order = df[hue].unique()
        n_hues = len(hue_order)
        bar_width = group_width / n_hues

        for i, cond in enumerate(cond_order):
            x_base = i
            for j, h in enumerate(hue_order):
                bar_center = (x_base - group_width / 2) + bar_width * (j + 0.5)
                subset = df[(df[x] == cond) & (df[hue] == h)][y]
                mean_val = subset.mean()
                se_val = subset.std() / np.sqrt(subset.count())
                bar_positions[(cond, h)] = (bar_center, mean_val + se_val)
    else:
        cond_order = df[x].unique()
        for i, cond in enumerate(cond_order):
            bar_center = i
            subset = df[df[x] == cond][y]
            mean_val = subset.mean()
            se_val = subset.std() / np.sqrt(subset.count())
            bar_positions[cond] = (bar_center, mean_val + se_val)

    return bar_positions

def add_significance_bars(fig, ax, df, x, y, hue=None, pairs=None, test=ttest_ind, alpha_levels=[0.05, 0.01, 0.001]):
    """
    adds significance markers between specified pairs to an existing barplot.

    Args:
        df (pd.DataFrame): Input data.
        x (str): Categorical variable for x-axis.
        y (str): Numeric variable for y-axis.
        hue (str, optional): Optional hue grouping.
        pairs (list of tuple): List of pairs to compare, each a tuple of category labels.
        test (function): Statistical test function taking two arrays.
        alpha_levels (list): Significance thresholds for stars.

    Returns:
        fig, ax: Matplotlib figure and axis objects.
    """
    bar_positions = compute_bar_positions_from_df(df, x, y, hue=hue)

    # If no pairs provided, compare all possible pairs
    if pairs is None:
        if hue:
            categories = df[[x, hue]].drop_duplicates().values.tolist()
            pairs = list(itertools.combinations(categories, 2))
        else:
            categories = df[x].unique()
            pairs = list(itertools.combinations(categories, 2))
    
    # Function to get star markers
    def pval_to_stars(p):
        if p < alpha_levels[2]:
            return '***'
        elif p < alpha_levels[1]:
            return '**'
        elif p < alpha_levels[0]:
            return '*'
        else:
            return 'ns'

    # Calculate max height for annotation
    if hue:
        group_stats = df.groupby([x, hue])[y].agg(['mean', 'count', 'std']).reset_index()
    else:
        group_stats = df.groupby(x)[y].agg(['mean', 'count', 'std']).reset_index()

    group_stats['se'] = group_stats['std'] / group_stats['count']**0.5
    max_height = (group_stats['mean'] + group_stats['se']).max()


    # Add significance markers
    y_offset = max_height * 0.2
    for i, pair in enumerate(pairs):
        if hue:
            (cat1, hue1), (cat2, hue2) = pair
            data1 = df[(df[x] == cat1) & (df[hue] == hue1)][y]
            data2 = df[(df[x] == cat2) & (df[hue] == hue2)][y]
            xpos1, height1 = bar_positions[(cat1, hue1)]
            xpos2, height2 = bar_positions[(cat2, hue2)]
        else:
            cat1, cat2 = pair
            data1 = df[df[x] == cat1][y]
            data2 = df[df[x] == cat2][y]
            xpos1, height1 = bar_positions[cat1]
            xpos2, height2 = bar_positions[cat2]

        stat, pval = test(data1, data2)
        stars = pval_to_stars(pval)

        # Draw line and text
        bar_y = max_height + y_offset * (i + 1.5)
        ax.plot([xpos1, xpos1, xpos2, xpos2], [bar_y, bar_y + y_offset * 0.5, bar_y + y_offset * 0.5, bar_y], color='black')
        ax.text((xpos1 + xpos2) / 2, bar_y + y_offset * 0.5, stars, ha='center', va='bottom')

    fig.tight_layout()
    return fig, ax