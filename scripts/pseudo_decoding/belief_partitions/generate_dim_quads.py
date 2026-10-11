"""
Finds the feature quadruples for dimension CCGP (dim_ccgp_feat_beliefs.py): X, X' in one dimension,
Y, Y' in another, with sessions where each of the four is the rule for at least BLOCK_THRESH
blocks, keeping quadruples with at least MIN_SESSIONS such sessions. Sessions are the valid
sessions of both subjects. Row order is stable, so a row's index is its --quad_idx.

    python3 scripts/pseudo_decoding/belief_partitions/generate_dim_quads.py
"""

import pandas as pd
import utils.behavioral_utils as behavioral_utils

QUADS_PATH = "/data/patrick_res/sessions/both/quads_at_least_2blocks_10sess.pickle"
BLOCK_THRESH = 2
MIN_SESSIONS = 10


def main():
    all_beh = pd.concat([behavioral_utils.load_all_beh_for_sub(sub) for sub in ["SA", "BL"]])
    quads = behavioral_utils.get_good_quads_across_sessions(all_beh, BLOCK_THRESH, MIN_SESSIONS)
    print(f"{len(quads)} quadruples, by dimension pair:")
    print(quads.groupby(["dim_a", "dim_b"]).size())
    print(f"sessions per quadruple: {quads.num_sessions.describe().to_dict()}")
    quads.to_pickle(QUADS_PATH)


if __name__ == "__main__":
    main()
