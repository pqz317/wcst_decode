"""
Given decoder results and shuffles, compute p values for both 
- decoding by time, 
- cross decoding by time. 
Store in similar fashion
"""

import os
import numpy as np
import pandas as pd
import utils.visualization_utils as visualization_utils

import utils.io_utils as io_utils

import utils.glm_utils as glm_utils
import utils.stats_utils as stats_utils
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
    ("both", None, None),
    ("both", "structure_level2_cleaned", "amygdala_Amy"),
    ("both", "structure_level2_cleaned", "basal_ganglia_BG"),
    ("both", "structure_level2_cleaned", "inferior_temporal_cortex_ITC"),
    ("both", "structure_level2_cleaned", "medial_pallium_MPal"),
    ("both", "structure_level2_cleaned", "lateral_prefrontal_cortex_lat_PFC"),
    ("both", "structure_level2_cleaned", "anterior_cingulate_gyrus_ACgG"),
    # ("SA", None, None),
    # ("BL", None, None),
    # ("SA", "drive", "Anterior"),
    # ("SA", "drive", "Temporal"),
]

DECODE_VARS = ["pref", "conf", "choice", "reward"]

TRIAL_EVENTS = ["StimOnset", "FeedbackOnsetLong"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(f'--combo_id', default=None, type=int)
    args = parser.parse_args()

    combos = list(itertools.product(SUB_REGION_LEVEL_REGIONS, DECODE_VARS, TRIAL_EVENTS))
    (sub, region_level, regions), decode_var, trial_event = combos[args.combo_id]
    print(f"computing p vals for {combos[args.combo_id]}")
    args = argparse.Namespace(
        **BeliefPartitionConfigs()._asdict()
    )
    args.subject = sub
    args.region_level = region_level
    args.regions = regions
    args.mode = decode_var
    args.trial_event = trial_event

    if decode_var in ["choice", "reward"]:
        args.base_output_path = "/data/patrick_res/choice_reward"
    else: 
        args.beh_filters = {"Response": "Correct", "Choice": "Chose"}

    if decode_var == "reward":
        args.sig_unit_level = "response_99th_window_filter_drift"
    else: 
        args.sig_unit_level = f"{decode_var}_99th_window_filter_drift"

    res = belief_partitions_io.read_results(args, FEATURES)
    shuffles = res[res["mode"] == f"{args.mode}_shuffle"]
    cross_res = belief_partitions_io.read_cross_time_results(args, FEATURES)

    p_vals = stats_utils.compute_p_for_decoding_by_time(res, args)

    cross_p_vals = stats_utils.compute_p_for_cross_decoding_by_time(cross_res, shuffles, args)

    args.model_trial_event = "StimOnset" if args.trial_event == "FeedbackOnsetLong" else "FeedbackOnsetLong"
    cross_event_res = belief_partitions_io.read_cross_time_results(args, FEATURES)
    cross_event_p_vals = stats_utils.compute_p_for_cross_decoding_by_time(cross_event_res, shuffles, args)

    out_dir = belief_partitions_io.get_dir_name(args)
    p_vals.to_pickle(os.path.join(out_dir, f"{args.mode}_pvals.pickle"))
    cross_p_vals.to_pickle(os.path.join(out_dir, f"{args.mode}_cross_p_vals.pickle"))
    cross_event_p_vals.to_pickle(os.path.join(out_dir, f"{args.mode}_{args.model_trial_event}_model_cross_p_vals.pickle"))

if __name__ == "__main__":
    main()