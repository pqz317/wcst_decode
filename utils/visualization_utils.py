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

REGION_TO_COLOR = {
    "Amygdala": "#00ff00",
    "Anterior Cingulate Gyrus": "#ff4500",
    "Basal Ganglia": "#0000ff",
    "Claustrum": "#ffd700",
    "Hippocampus/MTL": "#00ffff",
    "Parietal Cortex": "#191970",
    "Prefrontal Cortex": "#bc8f8f",
    "Premotor Cortex": "#006400",
    "Visual Cortex": "#ff1493",
}

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


def visualize_ccpg_value(args, df, ax):
    sns.lineplot(df, x="Time", y="Accuracy", hue="condition", linewidth=3, ax=ax)
    # # add estimated chance
    ax.axhline(1/2, color='black', linestyle='dotted', label="Estimated Chance")
    if args.trial_event == "FeedbackOnset":
        ax.axvspan(-0.8, 0, alpha=0.3, color='gray')
    ax.legend()
    ax.set_ylabel("Decoder Accuracy")
    ax.set_xlabel(f"Time Relative to {args.trial_event}")
    ax.set_title(f"Subject {args.subject} CCGP of value, {args.regions} regions")

def visualize_preferred_beliefs(args, df, ax, show_shuffles=True):
    if not show_shuffles:
        df = df[~df.condition.str.contains("shuffle")]
    sns.lineplot(df, x="Time", y="Accuracy", hue="condition", linewidth=3, ax=ax)
    # # add estimated chance
    ax.axhline(1/2, color='black', linestyle='dotted', label="Estimated Chance")
    if args.trial_event == "FeedbackOnset" or args.trial_event == "FeedbackOnsetLong":
        ax.axvspan(-0.8, 0, alpha=0.3, color='gray')
    ax.legend()
    ax.set_ylabel("Decoder Accuracy")
    ax.set_xlabel(f"Time Relative to {args.trial_event}")
    ax.set_title(f"Subject {args.subject} preferred beliefs, {args.regions} regions")