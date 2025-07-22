"""
Run through each of the decoding results computed, generate plots: 
- decoding accuracy by time
- cross decoding accuracy by time
Generate svg, png of each
Store in figures/wcst_paper/decoding
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

SUB_REGION_LEVEL_REGIONS = [
    ("both", "structure_level2_cleaned", "amygdala_Amy"),
    ("both", "structure_level2_cleaned", "basal_ganglia_BG"),
    ("both", "structure_level2_cleaned", "inferior_temporal_cortex_ITC"),
    ("both", "structure_level2_cleaned", "medial_pallium_MPal"),
    ("both", "structure_level2_cleaned", "lateral_prefrontal_cortex_lat_PFC"),
    ("SA", None, None),
    ("BL", None, None),
    ("both", None, None),
    ("SA", "drive", "Anterior"),
    ("SA", "drive", "Temporal"),
]

DECODE_VARS = ["pref", "conf", "choice", "reward"]

output_dir = "/data/patrick_res/figures/wcst_paper/decoding"

def plot_weights(args):
    all_conts = belief_partitions_io.get_contributions_for_all_time(args, region_level="structure_level2")
    peaks, orders = spike_utils.find_peaks(all_conts, value_col="mean_cont", time_col="abs_time", region_level="structure_level2")
    return visualization_utils.plot_pop_heatmap_by_time(all_conts, value_col="mean_cont", region_level="structure_level2", orders=orders)


def main():
    for (sub, region_level, regions), decode_var in tqdm(list(itertools.product(SUB_REGION_LEVEL_REGIONS, DECODE_VARS))):
        args = argparse.Namespace(
            **BeliefPartitionConfigs()._asdict()
        )
        args.subject = sub
        args.region_level = region_level
        args.regions = regions
        args.mode = decode_var

        if decode_var in ["choice", "reward"]:
            args.base_output_path = "/data/patrick_res/choice_reward"
        else: 
            args.beh_filters = {"Response": "Correct", "Choice": "Chose"}

        if decode_var == "reward":
            args.sig_unit_level = "response_99th_window_filter_drift"

        else: 
            args.sig_unit_level = f"{decode_var}_99th_window_filter_drift"
        try: 
            fig_acc, _ = visualization_utils.plot_combined_accs(args)
            fig_acc.savefig(f"{output_dir}/{sub}_{regions}_{decode_var}_accs.svg")
            fig_acc.savefig(f"{output_dir}/{sub}_{regions}_{decode_var}_accs.png")

        except Exception as e:
            print(e)

        try: 
            fig_cross, _ = visualization_utils.plot_combined_cross_accs(args)
            fig_cross.savefig(f"{output_dir}/{sub}_{regions}_{decode_var}_cross_time_accs.svg")
            fig_cross.savefig(f"{output_dir}/{sub}_{regions}_{decode_var}_cross_time_accs.png")
        except Exception as e:
            print(e)
        fig_weights, _ = plot_weights(args)
        fig_weights.savefig(f"{output_dir}/{sub}_{regions}_{decode_var}_weights.svg")
        fig_weights.savefig(f"{output_dir}/{sub}_{regions}_{decode_var}_weights.png")
if __name__ == "__main__":
    main()