"""
Finds the sessions for held-out feature dimension CCGP (dim_ccgp_heldout_feat_beliefs.py): for each
pair of dimensions, sessions where each of the two dimensions has at least MIN_FEATS distinct
features that were the rule for at least BLOCK_THRESH blocks. Features can differ across sessions.
Sessions are the valid sessions of both subjects. Row order is stable, so a row's index is its
--dim_pair_idx.

    python3 scripts/pseudo_decoding/belief_partitions/generate_dim_heldout_sessions.py
"""

import pandas as pd
import utils.behavioral_utils as behavioral_utils

DIM_PAIRS_PATH = "/data/patrick_res/sessions/both/dim_pairs_3feats_at_least_2blocks.pickle"
BLOCK_THRESH = 2
MIN_FEATS = 3


def main():
    all_beh = pd.concat([behavioral_utils.load_all_beh_for_sub(sub) for sub in ["SA", "BL"]])
    dim_pairs = behavioral_utils.get_good_dim_pairs_across_sessions(all_beh, BLOCK_THRESH, MIN_FEATS)
    print(dim_pairs[["dim_a", "dim_b", "num_sessions"]])
    dim_pairs.to_pickle(DIM_PAIRS_PATH)


if __name__ == "__main__":
    main()
