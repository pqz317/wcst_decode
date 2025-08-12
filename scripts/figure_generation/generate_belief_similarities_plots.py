"""
Generate pairs plots for averaged different in cosine similarity for pairs within vs across dimensions
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
    ("both", "structure_level2_cleaned", "anterior_cingulate_gyrus_ACgG"),
    # ("both", None, None),
]

BOTH_PAIRS_PATH = "/data/patrick_res/sessions/both/pairs_at_least_3blocks_10sess.pickle"


# TODO: move these functions somwhere else, just don't know where...
def compute_within_across_diff(sim_res):
    within_res = sim_res[sim_res.dim_type == "within dim"]
    across_res = sim_res[sim_res.dim_type == "across dim"]
    merged = pd.merge(within_res, across_res, how="cross", suffixes=("_within", "_across"))
    return pd.DataFrame({"TimeIdx": sim_res.name, "diff": merged["cosine_sim_within"] - merged["cosine_sim_across"]})

def get_trial_averaged_res(args, pairs):
    all_res = []
    for i, pair in pairs.iterrows():
        args.feat_pair = pair.pair
        out_dir = belief_partitions_io.get_dir_name(args, make_dir=False)
        file_name = belief_partitions_io.get_ccgp_file_name(args)
        res = pd.read_pickle(os.path.join(out_dir, f"{file_name}_{args.sim_type}.pickle"))
        res["dim_type"] = pair.dim_type
        res["pair_str"] = "_".join(pair.pair)
        all_res.append(res)
    all_res = pd.concat(all_res)
    trial_averaged = all_res.groupby(["TimeIdx", "pair_str", "dim_type"]).cosine_sim.mean().reset_index()
    return trial_averaged

def load_data(args, pairs):
    trial_averaged = get_trial_averaged_res(args, pairs)
    diffs = trial_averaged.groupby("TimeIdx").apply(compute_within_across_diff).reset_index(drop=True)

    all_shuffs = []
    for i in range(10):
        args.shuffle_idx = i
        shuff = get_trial_averaged_res(args, pairs)
        shuff["shuffle_idx"] = i
        all_shuffs.append(shuff)
    all_shuffs = pd.concat(all_shuffs)
    shuff_diffs = all_shuffs.groupby("TimeIdx").apply(compute_within_across_diff).reset_index(drop=True)

    diffs["shuffle_type"] = "true"
    shuff_diffs["shuffle_type"] = "shuffle"
    all_diffs = pd.concat((diffs, shuff_diffs))
    all_diffs["Time"] = all_diffs["TimeIdx"] / 10 + 0.1

    # time points of interest
    all_diffs = all_diffs[all_diffs.TimeIdx <= 0]
    return all_diffs

def main():
    plt.rcParams.update({'font.size': 16})
    pairs = pd.read_pickle(BOTH_PAIRS_PATH).reset_index(drop=True)
    for (sub, region_level, regions) in tqdm(SUB_REGION_LEVEL_REGIONS):
        pairs = pd.read_pickle(BOTH_PAIRS_PATH).reset_index(drop=True)
        args = argparse.Namespace(
            **BeliefPartitionConfigs()._asdict()
        )
        args.subject = "both"
        args.base_output_path = "/data/patrick_res/belief_similarities"
        args.sim_type = "cosine_sim"

        args.region_level = region_level
        args.regions = regions

        all_diffs = load_data(args, pairs)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), width_ratios=(4, 6), sharey=True)
        sns.barplot(all_diffs, x="shuffle_type", y="diff", errorbar="se", ax=ax1)
        visualization_utils.add_significance_bars(fig, ax1, all_diffs, "shuffle_type", "diff")

        sns.lineplot(all_diffs, x="Time", y="diff", hue="shuffle_type", errorbar="se", linewidth=3, ax=ax2)
        ax1.set_ylabel("sim(within) - sim(across)")
        ax1.set_xlabel("")
        ax2.set_xlabel("Time to cards appear")
        ax2.axvline(-0.5, color='grey', linestyle='dotted', linewidth=3)
        region_str = "Whole Population" if regions is None else regions
        # fig.suptitle(region_str)
        ax2.legend(title=None)
        for line in ax2.legend().get_lines():
            line.set_linewidth(6)

        visualization_utils.format_plot([ax1, ax2], axislabelsize=16)

        fig.tight_layout()

        output_dir = "/data/patrick_res/figures/wcst_paper/belief_similarities"
        fig.savefig(f"{output_dir}/{sub}_{regions}_{args.sim_type}.svg")
        fig.savefig(f"{output_dir}/{sub}_{regions}_{args.sim_type}.png")
    
if __name__ == "__main__":
    main()