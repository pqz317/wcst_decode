"""
Generate plots for updating projections for both pref and conf
Two plots: 
- one for 
"""

import os
import numpy as np
import pandas as pd
import utils.visualization_utils as visualization_utils

import utils.io_utils as io_utils

import utils.glm_utils as glm_utils
import utils.classifier_utils as classifier_utils
from matplotlib import pyplot as plt
import matplotlib
import utils.spike_utils as spike_utils
from constants.behavioral_constants import *
from constants.decoding_constants import *
from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import *
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
import itertools

import argparse
import copy
from tqdm import tqdm
import seaborn as sns

SUB_REGION_LEVEL_REGIONS = [
    # ("both", "structure_level2_cleaned", "amygdala_Amy"),
    # ("both", "structure_level2_cleaned", "basal_ganglia_BG"),
    # ("both", "structure_level2_cleaned", "inferior_temporal_cortex_ITC"),
    # ("both", "structure_level2_cleaned", "medial_pallium_MPal"),
    # ("both", "structure_level2_cleaned", "lateral_prefrontal_cortex_lat_PFC"),
    # ("both", None, None),
    ("SA", None, None),
]

OUTPUT_DIR = "/data/patrick_res/figures/wcst_paper/update_projections"

order= ["chose X / correct", "correct", "shuffle", "incorrect", "chose X / incorrect"]
conditions_maps = {
    "chose X / correct": {"Response": "Correct", "Choice": "Chose"},
    "chose X / incorrect": {"Response": "Incorrect", "Choice": "Chose"},

    "correct": {"Response": "Correct"},
    "incorrect": {"Response": "Incorrect"},
}

names_to_intervals = {
    "cross fixation": [(-1, 0, "StimOnset")],
    "decision": [(0, 1.1, "StimOnset"), (-1.8, -0.8, "FeedbackOnsetLong")],
    "card fixation": [(-0.8, 0, "FeedbackOnsetLong")],
    "feedback": [(0, 1.6, "FeedbackOnsetLong")],
}

def read_all_cond_data(args):
    all_data = []
    for i, event in enumerate(["StimOnset", "FeedbackOnsetLong"]):
        all_conds = []
        for cond_name in conditions_maps:
            args.trial_event = event
            args.conditions = conditions_maps[cond_name]
            res = belief_partitions_io.read_update_projections(args)
            res["cond"] = res.apply(lambda x: "shuffle" if "shuffle" in x["mode"] else cond_name, axis=1)
            all_conds.append(res)
        all_conds = pd.concat(all_conds)
        all_conds["trial_event"] = event
        all_conds["Time"] = all_conds["TimeIdx"] / 10 + 0.1
        all_data.append(all_conds)
    return pd.concat(all_data)

def plot_by_time(data):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), width_ratios=(20, 33), sharey=True)
    for ax_idx, event in enumerate(["StimOnset", "FeedbackOnsetLong"]):
        event_data = data[data.trial_event == event]
        sns.lineplot(event_data, x="Time", y="proj", hue="cond", ax=axs[ax_idx], hue_order=order)
        axs[ax_idx].set_xlabel(f"Time to {event}")
    axs[0].set_ylabel("Change in projection along preference of X")
    visualization_utils.format_plot(axs)
    fig.tight_layout()
    return fig, axs

def plot_by_intervals(data):
    fig, axs = plt.subplots(4, 1, figsize=(8, 20))
    for i, (interval_name, intervals) in enumerate(names_to_intervals.items()):
        sub_data = []
        for (pre, post, trial_event) in intervals:
            sub_data.append(data[(data.Time >= pre) & (data.Time < post) & (data.trial_event == trial_event)])
        sub_data = pd.concat(sub_data)
        sub_data = sub_data.sort_values(by="cond", key=lambda x: x.map(order.index))

        sns.barplot(sub_data, x="cond", y="proj", errorbar="se", ax=axs[i])
        visualization_utils.add_significance_bars(fig, axs[i], sub_data, "cond", "proj", pairs=[
            ("chose X / correct", "shuffle"),
            ("chose X / incorrect", "shuffle"),
            ("correct", "shuffle"),
            ("incorrect", "shuffle"),
        ])
        axs[i].set_ylabel("Change in projection along preference of X")
        axs[i].set_title(interval_name)
    visualization_utils.format_plot(axs)
    fig.tight_layout()
    return fig, axs


def plot_for_mode(args, mode):
    args.mode = mode
    args.sig_unit_level = f"{mode}_99th_window_filter_drift"
    data = read_all_cond_data(args)

    fig_by_time, _ = plot_by_time(data)
    fig_by_time.savefig(f"{OUTPUT_DIR}/{args.subject}_{args.regions}_{args.mode}_by_time.svg")
    fig_by_time.savefig(f"{OUTPUT_DIR}/{args.subject}_{args.regions}_{args.mode}_by_time.png")

    fig_by_intervals, _ = plot_by_intervals(data)
    fig_by_intervals.savefig(f"{OUTPUT_DIR}/{args.subject}_{args.regions}_{args.mode}_by_intervals.svg")
    fig_by_intervals.savefig(f"{OUTPUT_DIR}/{args.subject}_{args.regions}_{args.mode}_by_intervals.png")

def main():
    for (sub, region_level, regions) in tqdm(SUB_REGION_LEVEL_REGIONS):
        args = argparse.Namespace(
            **BeliefPartitionConfigs()._asdict()
        )
        args.subject = sub
        args.beh_filters = {"Response": "Correct", "Choice": "Chose"}

        args.region_level = region_level
        args.regions = regions
        args.base_output_path = "/data/patrick_res/update_projections"

        plot_for_mode(copy.deepcopy(args), "pref")
        plot_for_mode(copy.deepcopy(args), "conf")

        


if __name__ == "__main__":
    main()
