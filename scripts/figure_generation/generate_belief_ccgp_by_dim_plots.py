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

OUTPUT_DIR = "/data/patrick_res/figures/wcst_paper/ccgp_no_cond"

CCGP_DIM_TO_DISPLAY_NAMES = {
    "within dim": "within dim.",
    "within dim shuffle": "within dim. shuffle",
    "across dim": "across dim.",
    "across dim shuffle": "across dim. shuffle",
}

CCGP_DIM_TO_COLOR = {
    "within dim.": "#1b9e77",
    "within dim. shuffle": "#b3e2cd",
    "across dim.": "#d95f02",
    "across dim. shuffle": "#fdcdac"
}


def main():
    pairs = pd.read_pickle(PAIRS_PATH).reset_index(drop=True)
    # regions = [None] + REGIONS_OF_INTEREST
    regions = [None]
    for region in regions:
        args = argparse.Namespace(
            **BeliefPartitionConfigs()._asdict()
        )
        args.regions = region
        args.region_level = None if region is None else "structure_level2_cleaned"
        args.subject = "both"
        args.mode = "feat_belief"
        args.base_output_path = "/data/patrick_res/ccgp_conf"
        # args.sig_unit_level = "pref_conf_99th_window_filter_drift"
        args.sig_unit_level = "pref_conf_99th_no_cond_window_filter_drift"


        res = belief_partitions_io.read_ccgp_results(args, pairs, conds=["across_cond"])
        res = res[res.Time <= 0]
        res["condition"] = res.apply(lambda x: f"{x.dim_type} shuffle" if "shuffle" in x.condition else x.dim_type, axis=1)

        sig_pairs = [("within dim.", "across dim.", "black")]
        fig, (ax1, ax2) = visualization_utils.visualize_bars_time(args, res, 
            display_map=CCGP_DIM_TO_DISPLAY_NAMES, 
            color_map=CCGP_DIM_TO_COLOR,
            y_lims=(0.5, None),
            sig_pairs=sig_pairs
        )
        ax1.set_ylabel("Accuracy")
        fig.savefig(f"{OUTPUT_DIR}/{region}_belief_ccgp_by_dim.svg")
        fig.savefig(f"{OUTPUT_DIR}/{region}_belief_ccgp_by_dim.png", dpi=300)



if __name__ == "__main__":
    main()