"""
For each variable, generate decoding accuracies for each region combined into one plot
"""
import os
import numpy as np
import pandas as pd
import utils.visualization_utils as visualization_utils

import utils.io_utils as io_utils

import utils.glm_utils as glm_utils
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
import matplotlib.cm as cm
import matplotlib.colors as mcolors

DECODE_VARS = ["pref", "conf", "choice", "reward"]
REGIONS = ["medial_pallium_MPal", "inferior_temporal_cortex_ITC", "basal_ganglia_BG", "amygdala_Amy", "lateral_prefrontal_cortex_lat_PFC", "anterior_cingulate_gyrus_ACgG"]
OUTPUT_DIR = "/data/patrick_res/figures/wcst_paper/decoding_combined_region_updated"


def main():
    for decode_var in DECODE_VARS:
        args = argparse.Namespace(
            **BeliefPartitionConfigs()._asdict()
        )
        args.subject = "both"
        args.region_level="structure_level2_cleaned"
        args.mode = decode_var
        if decode_var in ["choice", "reward"]:
            args.base_output_path = "/data/patrick_res/choice_reward"
        else: 
            args.beh_filters = {"Response": "Correct", "Choice": "Chose"}

        if decode_var == "reward":
            args.sig_unit_level = "response_99th_window_filter_drift"
        else: 
            args.sig_unit_level = f"{decode_var}_99th_window_filter_drift"
        fig, axs = visualization_utils.plot_combined_accs_by_attr(
            args, 
            attr="regions", 
            values=REGIONS, 
            num_shuffles=10, 
            hue_order=visualization_utils.REGION_ORDER,
            palette=visualization_utils.REGION_TO_COLOR,
            display_names=visualization_utils.REGION_TO_DISPLAY_NAMES)
        fig.savefig(f"{OUTPUT_DIR}/{decode_var}_accs.svg")
        fig.savefig(f"{OUTPUT_DIR}/{decode_var}_accs.png")

if __name__ == "__main__":
    main()