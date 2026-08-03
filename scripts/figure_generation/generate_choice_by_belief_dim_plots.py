"""
For each region and trial event, generate choice decoding accuracies split by belief dim
partition, overlaid in one plot, with difference-of-differences significance bars:
    Delta = (true_In - shuffle_In) - (true_NotIn - shuffle_NotIn) > 0
p vals computed by scripts/pseudo_decoding/p_val_computations/compute_p_vals_for_dod.py,
read out of the "In X Dim" partition dir.
One figure per event, plus a combined figure with both events side by side.
Generate svg, png of each, store in figures/wcst_paper/choice_by_belief_dim
"""
import os
import numpy as np
import pandas as pd
import utils.visualization_utils as visualization_utils

import utils.io_utils as io_utils

from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
from constants.behavioral_constants import *
from constants.decoding_constants import *
from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import *
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
import itertools

import argparse
import copy
from tqdm import tqdm

OUTPUT_DIR = "/data/patrick_res/figures/wcst_paper/choice_by_belief_dim"

BASE_OUTPUT_PATH = "/data/patrick_res/choice_belief_dim"
SIG_UNIT_LEVEL = "choice_99th_window_filter_drift"
NUM_SHUFFLES = 10
PARTITIONS = ["Low", "In X Dim", "Not in X Dim"]
# partitions actually overlaid against each other, in order
OVERLAY_PARTITIONS = ["In X Dim", "Not in X Dim"]

TRIAL_EVENTS = ["StimOnset", "FeedbackOnsetLong"]

# partition names are also data keys (beh_filters values, dir names), only relabel for display
PARTITION_LABELS = {
    "In X Dim":     "High belief in dim",
    "Not in X Dim": "High belief in other dim",
}

# per event x axis formatting, same conventions as visualization_utils.plot_combined_accs
EVENT_TO_XLABEL = {
    "StimOnset": "Time to stimuli appear (s)",
    "FeedbackOnsetLong": "Time to feedback (s)",
}
EVENT_TO_TICKS = {
    "StimOnset": [-1.0, -0.5, 0.0, 0.5, 1.0],
    "FeedbackOnsetLong": [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
}
# dotted lines marking the preceding event, and the event itself
EVENT_TO_VLINES = {
    "StimOnset": [-0.5, 0],
    "FeedbackOnsetLong": [-0.8, 0],
}

# same region layout used by compute_p_vals_for_dod.py
SUB_REGION_LEVEL_REGIONS = [
    ("both", None, None),
    ("both", "structure_level2_cleaned", "amygdala_Amy"),
    ("both", "structure_level2_cleaned", "basal_ganglia_BG"),
    ("both", "structure_level2_cleaned", "inferior_temporal_cortex_ITC"),
    ("both", "structure_level2_cleaned", "medial_pallium_MPal"),
    ("both", "structure_level2_cleaned", "lateral_prefrontal_cortex_lat_PFC"),
    ("both", "structure_level2_cleaned", "anterior_cingulate_gyrus_ACgG"),
]

# Solid colors for real data; muted versions for corresponding shuffles
PARTITION_COLORS = {
    # "Low":          "#1f77b4",
    "In X Dim":     "#ff7f0e",
    "Not in X Dim": "#2ca02c",
}
PARTITION_SHUFFLE_COLORS = {
    # "Low":          "#aec7e8",
    "In X Dim":     "#ffbb78",
    "Not in X Dim": "#98df8a",
}


def load_partition_results(sub, region_level, regions, trial_event):
    all_res = {}
    for partition in PARTITIONS:
        args = argparse.Namespace(**BeliefPartitionConfigs()._asdict())
        args.mode = "choice"
        args.subject = sub
        args.trial_event = trial_event
        args.region_level = region_level
        args.regions = regions
        args.sig_unit_level = SIG_UNIT_LEVEL
        args.base_output_path = BASE_OUTPUT_PATH
        args.beh_filters = {"BeliefDimPartition": partition}
        res = belief_partitions_io.read_results(args, FEATURES, num_shuffles=NUM_SHUFFLES)
        all_res[partition] = (args, res)
    return all_res


def load_dod_pvals(all_res):
    """
    Load the difference-of-differences p-values (per timepoint) testing whether the "In X Dim"
    partition decodes reliably further above its own shuffle baseline than "Not in X Dim":
        Delta = (true_In - shuffle_In) - (true_NotIn - shuffle_NotIn) > 0
    Computed by scripts/pseudo_decoding/p_val_computations/compute_p_vals_for_dod.py and stored in
    the In-Dim partition dir. Returns a df[Time, TimeIdx, p, n_feat], or None if not yet computed.
    """
    args, _ = all_res["In X Dim"]
    dod_path = os.path.join(belief_partitions_io.get_dir_name(args, make_dir=False), f"{args.mode}_dod_pvals.pickle")
    if not os.path.exists(dod_path):
        print(f"Warning: dod p-vals not found at {dod_path}")
        return None
    return pd.read_pickle(dod_path)


def plot_side_by_side(all_res, label=""):
    trial_event = all_res[PARTITIONS[0]][0].trial_event
    ticks = EVENT_TO_TICKS[trial_event]
    fig, axs = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
    for i, partition in enumerate(PARTITIONS):
        args, res = all_res[partition]
        ax = axs[i]
        visualization_utils.visualize_preferred_beliefs(
            args, res.copy(), ax,
            p_vals=None,
            hue_col="mode",
            palette=visualization_utils.MODE_TO_COLOR,
        )
        for vline in EVENT_TO_VLINES[trial_event]:
            ax.axvline(vline, color="grey", linestyle="dotted", linewidth=3)
        ax.set_title(f"Belief Dim Partition: {partition}")
        ax.set_xlabel(EVENT_TO_XLABEL[trial_event])
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks)
        if i == 0:
            for line in ax.legend().get_lines():
                line.set_linewidth(6)
        else:
            ax.get_legend().remove()
            ax.set_ylabel("")
    visualization_utils.format_plot(list(axs))
    if label:
        fig.suptitle(label, y=1.02)
    fig.tight_layout()
    return fig


def num_timepoints(all_res):
    """Number of decoded timepoints for an event, used to width-scale combined panels"""
    return all_res[OVERLAY_PARTITIONS[0]][1].Time.nunique()


def draw_overlay(all_res, ax, p_vals=None, legend=True):
    """
    Draw the partition overlay for a single trial event onto ax: solid lines for real data,
    dashed lines for the corresponding shuffles, x axis formatted for that event.
    Shared by the single event and the combined event figures, so they can't drift apart.
    """
    trial_event = all_res[OVERLAY_PARTITIONS[0]][0].trial_event
    ticks = EVENT_TO_TICKS[trial_event]
    hue_order = [PARTITION_LABELS[partition] for partition in OVERLAY_PARTITIONS]
    real_palette = {PARTITION_LABELS[p]: c for p, c in PARTITION_COLORS.items()}
    shuffle_palette = {PARTITION_LABELS[p]: c for p, c in PARTITION_SHUFFLE_COLORS.items()}
    dfs_real, dfs_shuffle = [], []
    for partition in OVERLAY_PARTITIONS:
        _, res = all_res[partition]
        df = res.copy()
        df["partition"] = PARTITION_LABELS[partition]
        is_shuffle = df["mode"].str.contains("shuffle")
        dfs_real.append(df[~is_shuffle])
        dfs_shuffle.append(df[is_shuffle])
    combined_real = pd.concat(dfs_real)
    combined_shuffle = pd.concat(dfs_shuffle)

    sns.lineplot(
        combined_shuffle, x="Time", y="Accuracy",
        hue="partition", hue_order=hue_order,
        palette=shuffle_palette,
        linewidth=2, linestyle="dashed",
        errorbar="se", ax=ax, legend=False,
    )
    sns.lineplot(
        combined_real, x="Time", y="Accuracy",
        hue="partition", hue_order=hue_order,
        palette=real_palette,
        linewidth=3, errorbar="se", ax=ax,
        legend="auto" if legend else False,
    )
    # difference-of-differences significance: In X Dim > Not in X Dim (above each own baseline)
    if p_vals is not None:
        visualization_utils.plot_sig_bars(p_vals, 0.46, ax, color="black")
    ax.axhline(y=0.5, linestyle="dotted", color="black")
    for vline in EVENT_TO_VLINES[trial_event]:
        ax.axvline(vline, color="grey", linestyle="dotted", linewidth=3)
    ax.set_xlabel(EVENT_TO_XLABEL[trial_event])
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_ylim([0.44, 1.0])
    if legend:
        for line in ax.legend(title="Partition").get_lines():
            line.set_linewidth(6)
    return ax


def plot_overlay(all_res, p_vals=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    draw_overlay(all_res, ax, p_vals=p_vals)
    ax.set_ylabel("Accuracy")
    visualization_utils.format_plot([ax])
    fig.tight_layout()
    return fig


def plot_combined_overlay(all_res_by_event, p_vals_by_event=None):
    """
    Both trial events side by side in one figure, panels width-scaled by number of timepoints
    and sharing a y axis, following visualization_utils.plot_combined_accs
    """
    p_vals_by_event = p_vals_by_event or {}
    fig, axs = plt.subplots(
        1, len(TRIAL_EVENTS),
        figsize=(12, 4), sharey="row",
        width_ratios=[num_timepoints(all_res_by_event[event]) for event in TRIAL_EVENTS],
    )
    for i, (ax, event) in enumerate(zip(axs, TRIAL_EVENTS)):
        # legend only on the leftmost panel, lines are identical across panels
        draw_overlay(all_res_by_event[event], ax, p_vals=p_vals_by_event.get(event), legend=(i == 0))
        ax.set_ylabel("Accuracy" if i == 0 else "")
    visualization_utils.format_plot(list(axs))
    fig.tight_layout()
    return fig


def main():
    plt.rcParams.update({'font.size': 16})
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for (sub, region_level, regions) in tqdm(SUB_REGION_LEVEL_REGIONS):
        label = visualization_utils.REGION_TO_ABBREV.get(regions, "All Regions")
        all_res_by_event, dod_pvals_by_event = {}, {}
        for trial_event in TRIAL_EVENTS:
            print(f"=== {label}, {trial_event} ===")
            all_res = load_partition_results(sub, region_level, regions, trial_event)
            dod_pvals = load_dod_pvals(all_res)
            all_res_by_event[trial_event] = all_res
            dod_pvals_by_event[trial_event] = dod_pvals

            fig_overlay = plot_overlay(all_res, p_vals=dod_pvals)
            fig_overlay.savefig(f"{OUTPUT_DIR}/{sub}_{regions}_{trial_event}_choice_by_belief_dim_accs.svg")
            fig_overlay.savefig(f"{OUTPUT_DIR}/{sub}_{regions}_{trial_event}_choice_by_belief_dim_accs.png")
            plt.close(fig_overlay)

            # fig_side = plot_side_by_side(all_res, label=label)
            # fig_side.savefig(f"{OUTPUT_DIR}/{sub}_{regions}_{trial_event}_choice_by_belief_dim_accs_side_by_side.svg")
            # fig_side.savefig(f"{OUTPUT_DIR}/{sub}_{regions}_{trial_event}_choice_by_belief_dim_accs_side_by_side.png")
            # plt.close(fig_side)

        fig_combined = plot_combined_overlay(all_res_by_event, p_vals_by_event=dod_pvals_by_event)
        fig_combined.savefig(f"{OUTPUT_DIR}/{sub}_{regions}_combined_choice_by_belief_dim_accs.svg")
        fig_combined.savefig(f"{OUTPUT_DIR}/{sub}_{regions}_combined_choice_by_belief_dim_accs.png")
        plt.close(fig_combined)


if __name__ == "__main__":
    main()
