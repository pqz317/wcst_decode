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
enters through the mode name, see decode_pref_on_choice_axis.get_proj_mode. --b_split_half /
--axis_b_split_half likewise, for the B split runs -- the scored half names the directory and the
axis's half names the mode.

Combos with nothing on disk are skipped rather than raising, so a grid that was only run for some
of TRIAL_EVENTS needs no --combo_id: the B split runs are feedback aligned only, for instance. A
run that is present but missing features still raises, since that is a broken run rather than an
absent one.

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


def has_results(args):
    """
    Whether this combo's projection run is on disk at all.

    Only "nothing is here", so an event that was never run is skipped while a run that is present
    but incomplete still reaches load_df and raises.
    """
    dir = belief_partitions_io.get_dir_name(args, make_dir=False)
    if not os.path.isdir(dir):
        return False
    for feat in FEATURES:
        args.feat = feat
        if os.path.exists(os.path.join(dir, f"{belief_partitions_io.get_file_name(args)}_test_accs.npy")):
            return True
    return False


def run_combo(combo, axis_beh_filters, b_split_half=None, axis_b_split_half=None):
    (sub, region_level, regions), trial_event = combo
    print(f"computing p vals for {combo}")

    args = argparse.Namespace(**BeliefPartitionConfigs()._asdict())
    args.subject = sub
    args.region_level = region_level
    args.regions = regions
    args.mode = get_proj_mode(axis_beh_filters, axis_b_split_half)
    args.trial_event = trial_event
    args.beh_filters = BEH_FILTERS
    args.sig_unit_level = SIG_UNIT_LEVEL
    args.base_output_path = PROJ_OUTPUT_PATH
    # the scored half names the directory, the axis's half names the mode -- both must match what
    # the projection was run with, or read_results looks in the wrong place
    args.b_split_half = b_split_half

    if not has_results(args):
        print(f"  nothing on disk for {trial_event}, skipping")
        return False

    res = belief_partitions_io.read_results(args, FEATURES, num_shuffles=NUM_SHUFFLES)
    p_vals = stats_utils.compute_p_for_decoding_by_time(res, args)

    # read_results resets shuffle_idx, so this is the run dir rather than its shuffles/ subdir
    out_path = os.path.join(belief_partitions_io.get_dir_name(args), f"{args.mode}_pvals.pickle")
    print(f"storing p vals in {out_path}")
    p_vals.to_pickle(out_path)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(f'--combo_id', default=None, type=int)
    # must match what the projection was run with, see module docstring
    parser.add_argument(f'--axis_beh_filters', default={}, type=lambda x: json.loads(x))
    # halves of group B the projection was run with, if it was run with the split at all
    parser.add_argument(f'--b_split_half', default=None, type=int)
    parser.add_argument(f'--axis_b_split_half', default=None, type=int)
    args = parser.parse_args()

    combos = list(itertools.product(SUB_REGION_LEVEL_REGIONS, TRIAL_EVENTS))
    if args.combo_id is None:
        # no combo specified -> run all of them, each takes a few seconds. Combos with no results
        # on disk are skipped, so a grid run for only some of TRIAL_EVENTS needs no --combo_id
        ran = sum(run_combo(c, args.axis_beh_filters, args.b_split_half, args.axis_b_split_half) for c in tqdm(combos))
        print(f"wrote p vals for {ran} of {len(combos)} combos, skipped {len(combos) - ran} with no results")
    else:
        run_combo(combos[args.combo_id], args.axis_beh_filters, args.b_split_half, args.axis_b_split_half)


if __name__ == "__main__":
    main()
