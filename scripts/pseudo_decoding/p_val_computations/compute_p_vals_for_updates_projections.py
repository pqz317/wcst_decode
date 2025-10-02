"""
Given projection updates results and shuffles, compute p values values of interest relative to shuffle
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
from constants.update_projections_constants import *
from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import *
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
import itertools

import argparse
import copy
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(f'--combo_id', default=None, type=int)
    args = parser.parse_args()

    combos = list(itertools.product(REGIONS, AXIS_VARS, CONDITION_MAPS.keys()))
    region, axis_var, cond = combos[args.combo_id]
    print(f"computing p vals for combo id {combos[args.combo_id]}")
    args = argparse.Namespace(
        **BeliefPartitionConfigs()._asdict()
    )
    args.subject = "both"
    args.region_level = None if region is None else "structure_level2_cleaned"
    args.regions = region
    args.mode = axis_var
    args.trial_event = TRIAL_EVENT
    args.sig_unit_level = f"{args.mode}_99th_no_cond_window_filter_drift"
    args.base_output_path = "/data/patrick_res/update_projections"

    print("reading data")
    res = belief_partitions_io.read_update_projections_all_conds(args, CONDITION_MAPS)
    res = res[res.Time <0]
    print("computing p")
    p = stats_utils.compute_p_per_group(res, "proj", "cond", label_a=cond, label_b="shuffle", test_type="two_sided")

    args.beh_filters = CONDITION_MAPS[cond]
    dir = belief_partitions_io.get_dir_name(args)
    file_name = os.path.join(dir, f"{axis_var}_p_val.txt")
    print(f"got p of {p}")
    print(f"storing pval in path {file_name}")
    with open(file_name, 'w') as f:
        f.write(str(p))
    
if __name__ == "__main__":
    main()