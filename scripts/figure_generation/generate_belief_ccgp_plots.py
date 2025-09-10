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

OUTPUT_DIR = "/data/patrick_res/figures/wcst_paper/ccgp"


def main():
    pairs = pd.read_pickle(PAIRS_PATH).reset_index(drop=True)
    regions = [None] + REGIONS_OF_INTEREST
    for region in regions:
        args = argparse.Namespace(
            **BeliefPartitionConfigs()._asdict()
        )
        args.regions = region
        args.region_level = None if region is None else "structure_level2_cleaned"
        args.subject = "both"
        args.mode = "feat_belief"
        args.base_output_path = "/data/patrick_res/ccgp_conf"
        args.sig_unit_level = "pref_conf_99th_window_filter_drift"

        res = belief_partitions_io.read_ccgp_results(args, pairs, conds=["within_cond", "across_cond"])
        res = res[res.Time <= 0]
        sig_pairs = [("within cond.", "within cond. shuffle", "#1b9e77"), ("across cond.", "across cond. shuffle", "#d95f02")]
        fig, (ax1, ax2) = visualization_utils.visualize_bars_time(args, res, y_lims=(0.5, None), sig_pairs=sig_pairs)
        ax1.set_ylabel("Accuracy")
        fig.savefig(f"{OUTPUT_DIR}/{region}_belief_ccgp.svg")
        fig.savefig(f"{OUTPUT_DIR}/{region}_belief_ccgp.png", dpi=300)



if __name__ == "__main__":
    main()