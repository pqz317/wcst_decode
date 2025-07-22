"""
Generate pairs plots for both cosine similarity and CCGP for pairs of feature preferences
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
    ("both", "structure_level2_cleaned", "amygdala_Amy"),
    ("both", "structure_level2_cleaned", "basal_ganglia_BG"),
    ("both", "structure_level2_cleaned", "inferior_temporal_cortex_ITC"),
    ("both", "structure_level2_cleaned", "medial_pallium_MPal"),
    ("both", "structure_level2_cleaned", "lateral_prefrontal_cortex_lat_PFC"),
    ("both", None, None),
]
BOTH_PAIRS_PATH = "/data/patrick_res/sessions/both/pairs_at_least_3blocks_10sess.pickle"


output_dir = "/data/patrick_res/figures/wcst_paper/pairs_pref"

def plot_combined_ccgp(args, pairs):
    fig, axs = plt.subplots(3, 2, figsize=(15, 15), sharey='row', width_ratios=[20, 33])

    for col_idx, trial_event in enumerate(["StimOnset", "FeedbackOnsetLong"]):
        args.trial_event = trial_event

        across_dim_res = belief_partitions_io.read_ccgp_results(args, pairs[pairs.dim_type == "across dim"], conds=["across_cond"], num_shuffles=10)
        within_dim_res = belief_partitions_io.read_ccgp_results(args, pairs[pairs.dim_type == "within dim"], conds=["across_cond"], num_shuffles=10)

        res = pd.concat((across_dim_res, within_dim_res))

        visualization_utils.visualize_ccpg_value(args, res, axs[0, col_idx])
        visualization_utils.visualize_ccpg_value(args, within_dim_res, axs[1, col_idx])
        visualization_utils.visualize_ccpg_value(args, across_dim_res, axs[2, col_idx])
    axs[0, 0].set_title("All pairs")
    axs[1, 0].set_title("Within Dim")
    axs[2, 0].set_title("Across Dim")
    fig.savefig(f"{output_dir}/{args.subject}_{args.regions}_ccgp_pref_accs.svg")
    fig.savefig(f"{output_dir}/{args.subject}_{args.regions}_ccgp_pref_accs.png")

def plot_combined_cosine_sim(args, pairs):
    fig, axs = plt.subplots(3, 2, figsize=(15, 15), sharey='row', width_ratios=[20, 33])
    for col_idx, trial_event in enumerate(["StimOnset", "FeedbackOnsetLong"]):
        args.trial_event = trial_event

        true_res = classifier_utils.compute_cosine_sim_between_pairs_of_feats(args, pairs.pair, use_ccgp=True)
        true_res["shuffle_type"] = "true"
        shuffle_res = []
        for i in range(10):
            shuffle_args = copy.deepcopy(args)
            shuffle_args.shuffle_idx = i
            res = classifier_utils.compute_cosine_sim_between_pairs_of_feats(shuffle_args, pairs.pair, use_ccgp=True)
            res["shuffle_idx"] = i
            shuffle_res.append(res)
        shuffle_res = pd.concat(shuffle_res)  
        shuffle_res["shuffle_type"] = "shuffle"

        all_res = pd.concat((true_res, shuffle_res))
        sns.lineplot(all_res, x="Time", y="cosine_sim", hue="shuffle_type", errorbar="se", ax=axs[0, col_idx])
        sns.lineplot(all_res[all_res.dim_type == "within_dim"], x="Time", y="cosine_sim", hue="shuffle_type", errorbar="se", ax=axs[1, col_idx])
        sns.lineplot(all_res[all_res.dim_type == "across_dim"], x="Time", y="cosine_sim", hue="shuffle_type", errorbar="se", ax=axs[2, col_idx])
    axs[0, 0].set_title("All pairs")
    axs[1, 0].set_title("Within Dim")
    axs[2, 0].set_title("Across Dim")
    fig.savefig(f"{output_dir}/{args.subject}_{args.regions}_cosine_sim_pref.svg")
    fig.savefig(f"{output_dir}/{args.subject}_{args.regions}_cosine_sim_pref.png")
        


def main():
    pairs = pd.read_pickle(BOTH_PAIRS_PATH).reset_index(drop=True)
    for (sub, region_level, regions) in tqdm(SUB_REGION_LEVEL_REGIONS):
        args = argparse.Namespace(
            **BeliefPartitionConfigs()._asdict()
        )
        args.subject = sub
        args.beh_filters = {"Response": "Correct", "Choice": "Chose"}
        args.mode = "pref"
        args.sig_unit_level = "pref_99th_window_filter_drift"

        args.region_level = region_level
        args.regions = regions
        args.base_output_path = "/data/patrick_res/ccgp_pref_new"

        plot_combined_ccgp(args, pairs)
        plot_combined_cosine_sim(args, pairs)

if __name__ == "__main__":
    main()
