"""
Per-bin significance of held-out feature dimension CCGP against its session-permute shuffles:
    p = P(mean(true*) - mean(shuffle*) >= mean(true) - mean(shuffle))   [one sided permutation]
pooled over dimension pairs x folds x splits, the same test as compute_p_vals_for_dim_ccgp.py, for
ccgp_acc and train_acc, the held-out accuracy on the training roles.

Reads the tables scripts/pseudo_decoding/belief_partitions/dim_ccgp_heldout_feat_beliefs.py writes,
for every dimension pair of generate_dim_heldout_sessions.py, and writes
    dim_ccgp_heldout_p_vals.pickle   (columns measure, Time, TimeIdx, p)
to the true runs' dir. Runs locally. --sig_unit_level picks the runs of that unit preselection,
matching the launcher's flag; omit it for the all-unit runs.

    python3 scripts/pseudo_decoding/p_val_computations/compute_p_vals_for_dim_ccgp_heldout.py \
        --sig_unit_level dim_pair_99th_pre_stim_window_filter_drift
"""

import os
import argparse
import pandas as pd

from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import BeliefPartitionConfigs
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
from scripts.pseudo_decoding.belief_partitions.generate_dim_heldout_sessions import DIM_PAIRS_PATH
from scripts.pseudo_decoding.p_val_computations.compute_p_vals_for_dim_ccgp import compute_p_vals

BASE_OUTPUT_PATH = "/data/patrick_res/dim_ccgp_heldout"
NUM_SHUFFLES = 10
MEASURES = ["ccgp_acc", "train_acc"]


def main():
    parser = argparse.ArgumentParser()
    # unit preselection the runs used, as passed to the launcher; None for all units
    parser.add_argument("--sig_unit_level", default=None, type=str)
    cli_args = parser.parse_args()

    args = argparse.Namespace(**BeliefPartitionConfigs()._asdict())
    args.subject = "both"
    args.trial_event = "StimOnset"
    args.mode = "dim_ccgp_heldout"
    args.base_output_path = BASE_OUTPUT_PATH
    args.sig_unit_level = cli_args.sig_unit_level

    dim_pairs = pd.read_pickle(DIM_PAIRS_PATH)
    res = belief_partitions_io.read_dim_ccgp_heldout_results(args, dim_pairs, num_shuffles=NUM_SHUFFLES)
    p_vals = compute_p_vals(res, MEASURES)

    args.shuffle_idx = None
    out_path = os.path.join(belief_partitions_io.get_dir_name(args), "dim_ccgp_heldout_p_vals.pickle")
    print(f"storing p vals in {out_path}")
    p_vals.to_pickle(out_path)


if __name__ == "__main__":
    main()
