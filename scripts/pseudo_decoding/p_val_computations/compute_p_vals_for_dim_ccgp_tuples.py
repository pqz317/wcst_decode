"""
Per-bin significance of fixed-tuple dimension CCGP against its session-permute shuffles:
    p = P(mean(true*) - mean(shuffle*) >= mean(true) - mean(shuffle))   [one sided permutation]
pooled over tuples x folds x splits, separately for each fold type (dimension, random), for ccgp_acc
and train_acc, the held-out accuracy on the training features. The same test as
compute_p_vals_for_dim_ccgp_heldout.py.

Reads the tables scripts/pseudo_decoding/belief_partitions/dim_ccgp_tuple_feat_beliefs.py writes,
for every tuple of generate_dim_tuples.py, and writes
    dim_ccgp_tuples_p_vals.pickle   (columns fold_type, measure, Time, TimeIdx, p)
to the true runs' dir. Runs locally. The defaults pick the runs of slurm_launch_dim_ccgp_tuples_full.sh.

    python3 scripts/pseudo_decoding/p_val_computations/compute_p_vals_for_dim_ccgp_tuples.py --trial_event StimOnset
"""

import os
import json
import argparse
import pandas as pd

from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import BeliefPartitionConfigs
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
from scripts.pseudo_decoding.belief_partitions.generate_dim_tuples import TUPLES_PATH
from scripts.pseudo_decoding.belief_partitions.dim_ccgp_tuple_feat_beliefs import MODE, FOLD_TYPES
from scripts.pseudo_decoding.p_val_computations.compute_p_vals_for_dim_ccgp import compute_p_vals

BASE_OUTPUT_PATH = "/data/patrick_res/dim_ccgp_tuples"
SIG_UNIT_LEVEL = "pref_99th_dim_tuples_window_filter_drift"
BEH_FILTERS = {"Response": "Correct", "PreferredChosen": True}
NUM_SHUFFLES = 10
MEASURES = ["ccgp_acc", "train_acc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_event", default="StimOnset")
    parser.add_argument("--sig_unit_level", default=SIG_UNIT_LEVEL, type=str)
    parser.add_argument("--beh_filters", default=BEH_FILTERS, type=json.loads)
    cli_args = parser.parse_args()

    args = argparse.Namespace(**BeliefPartitionConfigs()._asdict())
    args.subject = "both"
    args.trial_event = cli_args.trial_event
    args.beh_filters = cli_args.beh_filters
    args.sig_unit_level = cli_args.sig_unit_level
    args.mode = MODE
    args.base_output_path = BASE_OUTPUT_PATH

    tuples = pd.read_pickle(TUPLES_PATH)
    res = belief_partitions_io.read_dim_ccgp_tuple_results(args, tuples, num_shuffles=NUM_SHUFFLES)
    p_vals = []
    for fold_type in FOLD_TYPES:
        fold_p_vals = compute_p_vals(res[res.fold_type == fold_type], MEASURES)
        fold_p_vals.insert(0, "fold_type", fold_type)
        p_vals.append(fold_p_vals)
    p_vals = pd.concat(p_vals, ignore_index=True)

    args.shuffle_idx = None
    out_path = os.path.join(belief_partitions_io.get_dir_name(args), "dim_ccgp_tuples_p_vals.pickle")
    print(f"storing p vals in {out_path}")
    p_vals.to_pickle(out_path)


if __name__ == "__main__":
    main()
