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

BOTH_PAIRS_PATH = "/data/patrick_res/sessions/both/pairs_at_least_3blocks_10sess.pickle"
OUTPUT_DIR = "/data/patrick_res/figures/wcst_paper/belief_similarities_explore_subpops_frs"

def main():
    plt.rcParams.update({'font.size': 16})
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(BeliefPartitionConfigs(), parser)
    parser.add_argument(f'--sim_type', default="cosine_sim")
    parser.add_argument(f'--relative_to_low', default=True, type=lambda x: bool(strtobool(x)))
    args = parser.parse_args()

    regions = [None] + REGIONS_OF_INTEREST
    for region in tqdm(regions):
        pairs = pd.read_pickle(BOTH_PAIRS_PATH).reset_index(drop=True)
        args.subject = "both"
        args.base_output_path = "/data/patrick_res/belief_similarities"
        # TODO: remove
        # args.sig_unit_level = "pref_conf_99th_no_cond_window_filter_drift"
        # args.sig_unit_level = "pref_conf_99th_window_filter_drift"
        args.region_level = None if region is None else "structure_level2_cleaned"
        args.regions = region

        all_data = belief_partitions_io.read_all_similarities(args, pairs)
        all_data = all_data[all_data.Time <= 0]
        within_data = all_data[all_data.dim_type == "within dim"]
        across_data = all_data[all_data.dim_type == "across dim"]

        crossed = pd.merge(within_data, across_data, on=["PseudoTrialNumber", "TimeIdx", "type"], suffixes=["_within", "_across"])
        crossed["sim_diff"] = crossed["cosine_sim_within"] - crossed["cosine_sim_across"]
        crossed["Time"] = crossed["TimeIdx"] / 10

        color_map = {
            "true": "tab:blue",
            "shuffle": "grey",
        }
        sig_pairs = [("true", "shuffle", "black")]
        fig, (ax1, ax2) = visualization_utils.visualize_bars_time(
            args, crossed, y_col="sim_diff", hue_col="type", 
            display_map=None, color_map=color_map, 
            y_lims=(None, None),
            sig_pairs=sig_pairs
        )
        ax1.set_ylabel("sim(within dim.) - sim(across dim.)")

        fig.savefig(f"{OUTPUT_DIR}/{region}_{args.sig_unit_level}_sim_diff_by_dim.svg")
        fig.savefig(f"{OUTPUT_DIR}/{region}_{args.sig_unit_level}sim_diff_by_dim.png", dpi=300)
    
if __name__ == "__main__":
    main()