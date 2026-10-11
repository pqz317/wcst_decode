"""
Per-bin significance of dimension CCGP against its session-permute shuffles:
    p = P(mean(true*) - mean(shuffle*) >= mean(true) - mean(shuffle))   [one sided permutation]
pooled over quadruples x assignments x splits, the same test compute_p_vals_for_decoders.py applies
to the pref decoders (stats_utils.compute_p_for_decoding_by_time), over the pre-stimulus bins only.
Also run on train_pair_acc, each decoder's held-out accuracy on its own train pair.

Reads the tables scripts/pseudo_decoding/belief_partitions/dim_ccgp_feat_beliefs.py writes, for every
quadruple of generate_dim_quads.py, and writes
    dim_ccgp_p_vals.pickle   (columns measure, Time, TimeIdx, p)
to the true runs' dir. Runs locally in a few minutes.
"""

import os
import argparse
import pandas as pd

import utils.stats_utils as stats_utils
from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import BeliefPartitionConfigs
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
from scripts.pseudo_decoding.belief_partitions.generate_dim_quads import QUADS_PATH

BASE_OUTPUT_PATH = "/data/patrick_res/dim_ccgp"
SIG_UNIT_LEVEL = "pref_99th_no_cond_window_filter_drift"
NUM_SHUFFLES = 10
MEASURES = ["ccgp_acc", "train_pair_acc"]


def compute_p_vals(res):
    p_res = []
    for measure in MEASURES:
        for (time_idx, time), time_res in res.groupby(["TimeIdx", "Time"]):
            p = stats_utils.compute_p_per_group(time_res, measure, "shuffle_type")
            p_res.append({"measure": measure, "Time": time, "TimeIdx": time_idx, "p": p})
    return pd.DataFrame(p_res)


def main():
    args = argparse.Namespace(**BeliefPartitionConfigs()._asdict())
    args.subject = "both"
    args.trial_event = "StimOnset"
    args.mode = "dim_ccgp"
    args.sig_unit_level = SIG_UNIT_LEVEL
    args.base_output_path = BASE_OUTPUT_PATH

    quads = pd.read_pickle(QUADS_PATH)
    res = belief_partitions_io.read_dim_ccgp_results(args, quads, num_shuffles=NUM_SHUFFLES)
    p_vals = compute_p_vals(res)

    args.shuffle_idx = None
    out_path = os.path.join(belief_partitions_io.get_dir_name(args), "dim_ccgp_p_vals.pickle")
    print(f"storing p vals in {out_path}")
    p_vals.to_pickle(out_path)


if __name__ == "__main__":
    main()
