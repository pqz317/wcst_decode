"""
Finds the feature 6-tuples for dimension CCGP on fixed tuples (dim_ccgp_tuple_feat_beliefs.py):
three features of one dimension and three of another, kept if every feature was the rule for at
least BLOCK_THRESH blocks in at least MIN_SESSIONS sessions. Sessions are the valid sessions of both
subjects. Rows are sorted by session count, descending and stable, so a row's index is its
--tuple_idx. Proposal: claude_notes/dimension_ccgp_fixed_tuples_proposal.md

Also writes, per subject, the (feature, session) pairs the tuples need, in the shape of
feats_at_least_3blocks.pickle, for run_anova2.py --feats_path, which selects the tuples' units.

    python3 scripts/pseudo_decoding/belief_partitions/generate_dim_tuples.py
"""

import itertools
import pandas as pd
import utils.behavioral_utils as behavioral_utils
from constants.behavioral_constants import FEATURES, FEATURE_TO_DIM

TUPLES_PATH = "/data/patrick_res/sessions/both/dim_tuples_3feats_at_least_2blocks.pickle"
TUPLE_FEATS_PATH = "/data/patrick_res/sessions/{sub}/feats_dim_tuples.pickle"
BLOCK_THRESH = 2
FEATS_PER_DIM = 3
MIN_SESSIONS = 7
SUBJECTS = ["SA", "BL"]


def get_tuples(num_blocks):
    """
    Every tuple of FEATS_PER_DIM features from each of two dimensions, with the sessions where all
    its features were the rule for >= BLOCK_THRESH blocks
    """
    good = num_blocks >= BLOCK_THRESH
    dims = sorted(set(FEATURE_TO_DIM.values()))
    rows = []
    for dim_a, dim_b in itertools.combinations(dims, 2):
        feats = {dim: [f for f in FEATURES if FEATURE_TO_DIM[f] == dim] for dim in [dim_a, dim_b]}
        for feats_a in itertools.combinations(feats[dim_a], FEATS_PER_DIM):
            for feats_b in itertools.combinations(feats[dim_b], FEATS_PER_DIM):
                feat_tuple = list(feats_a + feats_b)
                sessions = good.index[good[feat_tuple].all(axis=1)].values
                rows.append({
                    "dim_a": dim_a, "dim_b": dim_b, "feats_a": list(feats_a), "feats_b": list(feats_b),
                    "feat_tuple": feat_tuple, "sessions": sessions, "num_sessions": len(sessions),
                })
    return pd.DataFrame(rows)


def get_tuple_feats(tuples, sub_sessions):
    """
    Per feature, the sessions of sub_sessions where some tuple needs it, in the shape of
    feats_at_least_3blocks.pickle
    """
    rows = []
    for feat in FEATURES:
        sessions = set()
        for _, row in tuples[tuples.feat_tuple.apply(lambda t: feat in t)].iterrows():
            sessions |= set(row.sessions) & sub_sessions
        rows.append({"feat": feat, "sessions": sorted(sessions), "num_sessions": len(sessions)})
    return pd.DataFrame(rows)


def main():
    behs = {sub: behavioral_utils.load_all_beh_for_sub(sub) for sub in SUBJECTS}
    all_beh = pd.concat(behs.values())
    num_blocks = all_beh.groupby(["session", "CurrentRule"]).BlockNumber.nunique().unstack(fill_value=0)
    num_blocks = num_blocks.reindex(columns=FEATURES, fill_value=0)

    tuples = get_tuples(num_blocks)
    tuples = tuples[tuples.num_sessions >= MIN_SESSIONS]
    tuples = tuples.sort_values("num_sessions", ascending=False, kind="stable").reset_index(drop=True)
    print(tuples[["dim_a", "dim_b", "feat_tuple", "num_sessions"]].to_string())
    tuples.to_pickle(TUPLES_PATH)

    for sub, beh in behs.items():
        tuple_feats = get_tuple_feats(tuples, set(beh.session.unique()))
        print(f"{sub}: {tuple_feats.num_sessions.sum()} (feature, session) pairs")
        tuple_feats.to_pickle(TUPLE_FEATS_PATH.format(sub=sub))


if __name__ == "__main__":
    main()
