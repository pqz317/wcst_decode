import os
import numpy as np
import pandas as pd
import utils.behavioral_utils as behavioral_utils
import utils.information_utils as information_utils
import utils.visualization_utils as visualization_utils
import utils.pseudo_classifier_utils as pseudo_classifier_utils
import utils.classifier_utils as classifier_utils

import utils.io_utils as io_utils

import utils.glm_utils as glm_utils
from matplotlib import pyplot as plt
import matplotlib
import utils.spike_utils as spike_utils
import utils.subspace_utils as subspace_utils
import utils.stats_utils as stats_utils
from trial_splitters.condition_trial_splitter import ConditionTrialSplitter 
from utils.session_data import SessionData
from constants.behavioral_constants import *
from constants.decoding_constants import *
import seaborn as sns
from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import *
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io

import scipy
import argparse
import copy
from tqdm import tqdm

OUTPUT_DIR = "/data/patrick_res/figures/wcst_paper/belief_similarities_explore_subpops_frs"

def main():
    pairs = pd.read_pickle(PAIRS_PATH).reset_index(drop=True)
    regions = [None] + REGIONS_OF_INTEREST
    # regions = [None]
    # regions = ["anterior_cingulate_gyrus_ACgG"]
    # regions = ["lateral_prefrontal_cortex_lat_PFC"]
    for region in tqdm(regions):
        args = argparse.Namespace(
            **BeliefPartitionConfigs()._asdict()
        )
        args.regions = region
        args.region_level = None if region is None else "structure_level2_cleaned"
        args.subject = "both"
        args.base_output_path = "/data/patrick_res/belief_similarities"
        # TODO: remove
        # args.sig_unit_level = "pref_conf_99th_no_cond_window_filter_drift"
        args.relative_to_low = True
        args.sim_type = "cosine_sim"

        all_data = belief_partitions_io.read_all_similarities(args, pairs)
        # print("done reading...")
        all_data = all_data[all_data.Time <= 0]

        color_map = {
            "true": "tab:blue",
            "shuffle": "grey",
        }
        min = all_data.groupby(["TimeIdx", "type"]).cosine_sim.mean().min()
        sig_pairs = [("true", "shuffle", "black")]

        fig, (ax1, ax2) = visualization_utils.visualize_bars_time(
            args, all_data, y_col="cosine_sim", hue_col="type", 
            display_map=None, color_map=color_map, 
            y_lims=(min, None),
            sig_pairs=sig_pairs
        )

        ax1.set_ylabel("Cosine Sim")
        fig.savefig(f"{OUTPUT_DIR}/{region}_{args.sig_unit_level}_sim.svg")
        fig.savefig(f"{OUTPUT_DIR}/{region}_{args.sig_unit_level}_sim.png", dpi=300)



if __name__ == "__main__":
    main()