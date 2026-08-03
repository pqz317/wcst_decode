"""
Difference-of-differences significance test for choice decoding split by belief-dim partition.

For each region/event, tests per timepoint whether the "In X Dim" partition decodes reliably
further above its own shuffle baseline than the "Not in X Dim" partition does above its own
baseline:
    Delta = (true_In - shuffle_In) - (true_NotIn - shuffle_NotIn) > 0
See stats_utils.compute_p_for_dod_by_time for the null model (per-feature sign-flip, exact).

Reads the per-partition choice decoding results under /data/patrick_res/choice_belief_dim, and
writes the resulting per-time p-values to the "In X Dim" partition dir as
    {mode}_dod_pvals.pickle
(a deterministic, predictable path for later notebook loading).
"""

import os
import numpy as np
import pandas as pd

import utils.stats_utils as stats_utils
from constants.behavioral_constants import *
from constants.decoding_constants import *
from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import *
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
import itertools

import argparse
import copy
from tqdm import tqdm

BASE_OUTPUT_PATH = "/data/patrick_res/choice_belief_dim"
SIG_UNIT_LEVEL = "choice_99th_window_filter_drift"
NUM_SHUFFLES = 10

# same region layout used by compute_p_vals_for_decoders.py
SUB_REGION_LEVEL_REGIONS = [
    ("both", None, None),
    ("both", "structure_level2_cleaned", "amygdala_Amy"),
    ("both", "structure_level2_cleaned", "basal_ganglia_BG"),
    ("both", "structure_level2_cleaned", "inferior_temporal_cortex_ITC"),
    ("both", "structure_level2_cleaned", "medial_pallium_MPal"),
    ("both", "structure_level2_cleaned", "lateral_prefrontal_cortex_lat_PFC"),
    ("both", "structure_level2_cleaned", "anterior_cingulate_gyrus_ACgG"),
]

TRIAL_EVENTS = ["StimOnset", "FeedbackOnsetLong"]

IN_PARTITION = "In X Dim"
NOTIN_PARTITION = "Not in X Dim"


def read_partition_results(args, partition):
    part_args = copy.deepcopy(args)
    part_args.beh_filters = {"BeliefDimPartition": partition}
    return belief_partitions_io.read_results(part_args, FEATURES, num_shuffles=NUM_SHUFFLES)


def run_combo(combo):
    (sub, region_level, regions), trial_event = combo
    print(f"computing difference-of-differences p vals for {combo}")

    args = argparse.Namespace(**BeliefPartitionConfigs()._asdict())
    args.subject = sub
    args.region_level = region_level
    args.regions = regions
    args.mode = "choice"
    args.trial_event = trial_event
    args.sig_unit_level = SIG_UNIT_LEVEL
    args.base_output_path = BASE_OUTPUT_PATH

    res_in = read_partition_results(args, IN_PARTITION)
    res_notin = read_partition_results(args, NOTIN_PARTITION)

    p_vals = stats_utils.compute_p_for_dod_by_time(res_in, res_notin, args)

    # save to the In-Dim partition dir (deterministic path)
    out_args = copy.deepcopy(args)
    out_args.beh_filters = {"BeliefDimPartition": IN_PARTITION}
    out_args.shuffle_idx = None
    out_dir = belief_partitions_io.get_dir_name(out_args)
    out_path = os.path.join(out_dir, f"{args.mode}_dod_pvals.pickle")
    print(f"storing dod p vals in {out_path}")
    p_vals.to_pickle(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(f'--combo_id', default=None, type=int)
    args = parser.parse_args()

    combos = list(itertools.product(SUB_REGION_LEVEL_REGIONS, TRIAL_EVENTS))
    if args.combo_id is None:
        # no combo specified -> run all of them
        for combo in tqdm(combos):
            run_combo(combo)
    else:
        run_combo(combos[args.combo_id])


if __name__ == "__main__":
    main()
