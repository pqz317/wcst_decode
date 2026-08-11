"""
Significance of preference activity projected onto the choice axis, per timepoint.

For each region/event, tests whether accuracy along the choice axis sits above its own
session-permute shuffle:
    p = P(mean(true*) - mean(shuffle*) >= mean(true) - mean(shuffle))   [one sided permutation]
same test compute_p_vals_for_decoders.py applies to the decoders, see
stats_utils.compute_p_for_decoding_by_time.

Reads the projection results written by
scripts/pseudo_decoding/belief_partitions/decode_pref_on_choice_axis.py, and writes the per-time
p-values back into each run dir as
    {proj_mode}_pvals.pickle
matching the naming plot_combined_accs expects, so plot_sig_bars can pick them up directly.

--axis_beh_filters selects which projection to test, and must match what the projection was run
with: {} for the all-trials choice axis, {"Response": "Correct"} for the correct-only one. It only
enters through the mode name, see decode_pref_on_choice_axis.get_proj_mode.

Only decoding-by-time p values are computed here: the projection has no cross-time results, since
the axis it projects onto is only ever the one fit at the same timepoint.
"""

import os
import numpy as np
import pandas as pd

import utils.stats_utils as stats_utils
from constants.behavioral_constants import *
from constants.decoding_constants import *
from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import *
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
from scripts.pseudo_decoding.belief_partitions.decode_pref_on_choice_axis import get_proj_mode, PROJ_OUTPUT_PATH
import itertools

import argparse
import copy
import json
from tqdm import tqdm

# units, trials the projection reuses from the preference runs, same as slurm_launch_pref_on_choice_axis.sh
SIG_UNIT_LEVEL = "pref_99th_window_filter_drift"
BEH_FILTERS = {"Response": "Correct", "Choice": "Chose"}
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


def run_combo(combo, axis_beh_filters):
    (sub, region_level, regions), trial_event = combo
    print(f"computing p vals for {combo}")

    args = argparse.Namespace(**BeliefPartitionConfigs()._asdict())
    args.subject = sub
    args.region_level = region_level
    args.regions = regions
    args.mode = get_proj_mode(axis_beh_filters)
    args.trial_event = trial_event
    args.beh_filters = BEH_FILTERS
    args.sig_unit_level = SIG_UNIT_LEVEL
    args.base_output_path = PROJ_OUTPUT_PATH

    res = belief_partitions_io.read_results(args, FEATURES, num_shuffles=NUM_SHUFFLES)
    p_vals = stats_utils.compute_p_for_decoding_by_time(res, args)

    # read_results resets shuffle_idx, so this is the run dir rather than its shuffles/ subdir
    out_path = os.path.join(belief_partitions_io.get_dir_name(args), f"{args.mode}_pvals.pickle")
    print(f"storing p vals in {out_path}")
    p_vals.to_pickle(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(f'--combo_id', default=None, type=int)
    # must match what the projection was run with, see module docstring
    parser.add_argument(f'--axis_beh_filters', default={}, type=lambda x: json.loads(x))
    args = parser.parse_args()

    combos = list(itertools.product(SUB_REGION_LEVEL_REGIONS, TRIAL_EVENTS))
    if args.combo_id is None:
        # no combo specified -> run all of them, each takes a few seconds
        for combo in tqdm(combos):
            run_combo(combo, args.axis_beh_filters)
    else:
        run_combo(combos[args.combo_id], args.axis_beh_filters)


if __name__ == "__main__":
    main()
