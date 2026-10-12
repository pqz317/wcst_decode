"""
Dimension CCGP of feature beliefs with held-out features, for one pair of dimensions (A, B) from
generate_dim_heldout_sessions.py. Proposal: claude_notes/dimension_ccgp_heldout_features_proposal.md

Each session contributes three features per dimension that were the rule for >= 2 blocks, chosen at
random when it has four, and assigned at random to the roles A1-A3, B1-B3. Features differ across
sessions, so conditions are the roles, which keeps pseudo trials aligned across sessions. Each
session's High trials (PreferredBelief == f, PreferredBeliefProb > BELIEF_PARTITION_THRESH) are
balanced to the minimum count over its six features.

Fold k (0-2) holds out (A{k+1}, B{k+1}): a decoder labelled by dimension is trained on the other two
roles of each dimension, and tested on all trials of the held-out pair. Decoding is over the
pre-stimulus bins, [-1, 0]s around StimOnset, on all units.

With --shuffle_idx set, each session's behavior is replaced by a donor's from the dimension pair's
sessions, truncated to the session's length. A donor without three features per dimension with >= 2
blocks and >= 2 High trials is redrawn, up to MAX_DONOR_DRAWS donors, after which the session is
dropped from that run and counted in n_dropped_sessions.

Writes per run {stem}.pickle, one row per (fold, split, bin), with the held-out accuracy on the
training roles (train_acc) and the accuracy on the held-out pair (ccgp_acc), and {stem}_roles.pickle,
each kept session's role -> feature mapping, balanced trial count and donor. True runs also write
{stem}_fold_{k}_models.npy per fold (bins x splits) and {stem}_unit_ids.csv, the PseudoUnitIDs in
the decoder's input order.
"""

import os
import copy
import argparse
import numpy as np
import pandas as pd

import utils.behavioral_utils as behavioral_utils
import utils.spike_utils as spike_utils
import utils.pseudo_classifier_utils as pseudo_classifier_utils
from constants.behavioral_constants import *
from constants.decoding_constants import *

from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import BeliefPartitionConfigs, add_defaults_to_parser
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
from scripts.pseudo_decoding.belief_partitions.dim_ccgp_feat_beliefs import (
    DATA_MODE, PRE_STIM_RANGE, MIN_TRIALS_PER_COND, make_sess_datas, train_dim_decoder
)
from scripts.pseudo_decoding.belief_partitions.generate_dim_heldout_sessions import DIM_PAIRS_PATH, BLOCK_THRESH, MIN_FEATS

MODE = "dim_ccgp_heldout"
ROLES = [f"{prefix}{i + 1}" for prefix in ["A", "B"] for i in range(MIN_FEATS)]
MAX_DONOR_DRAWS = 10


def get_qualifying_feats(beh, dim):
    """
    Features of dim that were the rule for >= BLOCK_THRESH blocks and have >= MIN_TRIALS_PER_COND
    High trials, in FEATURES order
    """
    num_blocks = beh.groupby("CurrentRule").BlockNumber.nunique()
    num_high = beh[beh.PreferredBeliefProb > BELIEF_PARTITION_THRESH].PreferredBelief.value_counts()
    return [
        f for f in FEATURES
        if FEATURE_TO_DIM[f] == dim and num_blocks.get(f, 0) >= BLOCK_THRESH and num_high.get(f, 0) >= MIN_TRIALS_PER_COND
    ]


def is_qualifying(beh, args):
    return all(len(get_qualifying_feats(beh, dim)) >= MIN_FEATS for dim in [args.dim_a, args.dim_b])


def load_beh(sess_name, args):
    """
    The session's behavior, or in a shuffle run, that of the first qualifying donor out of
    MAX_DONOR_DRAWS. Returns (beh, donor_session, donor_draws); beh is None if no donor qualified,
    donor fields are NaN for true runs.
    """
    sub_args = copy.deepcopy(args)
    sub_args.subject = behavioral_utils.get_sub_for_session(sess_name)
    sub_args.shuffle_idx = None
    beh = behavioral_utils.load_behavior_from_args(sess_name, sub_args)
    if args.shuffle_idx is None:
        return beh, np.nan, np.nan

    others = [s for s in args.all_sessions.session_name if s != sess_name]
    rng = np.random.default_rng(int(sess_name) * 100 + args.shuffle_idx)
    for draw, donor in enumerate(rng.permutation(others)[:MAX_DONOR_DRAWS]):
        donor_beh = behavioral_utils.get_session_permuted_beh(beh, donor)
        if is_qualifying(donor_beh, args):
            return donor_beh, donor, draw + 1
    return None, np.nan, MAX_DONOR_DRAWS


def assign_roles(beh, sess_name, args):
    """
    Randomly chooses MIN_FEATS qualifying features per dimension and assigns them to roles,
    returns {role: feature}
    """
    shuffle_seed = 0 if args.shuffle_idx is None else args.shuffle_idx + 1
    rng = np.random.default_rng([args.train_test_seed, int(sess_name), shuffle_seed])
    role_to_feat = {}
    for prefix, dim in [("A", args.dim_a), ("B", args.dim_b)]:
        chosen = rng.permutation(get_qualifying_feats(beh, dim))[:MIN_FEATS]
        for i, feat in enumerate(chosen):
            role_to_feat[f"{prefix}{i + 1}"] = feat
    return role_to_feat


def load_session(row, args):
    """
    Loads a session's balanced High trials of its six chosen features, condition = role, and its
    firing rates. Returns (session dict or None, dropped), where dropped marks a session left out
    for lacking qualifying features, as opposed to having no units.
    """
    sess_name = row.session_name
    sub_args = copy.deepcopy(args)
    sub_args.subject = behavioral_utils.get_sub_for_session(sess_name)

    frs = spike_utils.get_frs_from_args(sub_args, sess_name)
    frs = frs.rename(columns={DATA_MODE: "Value"})
    if len(frs) == 0:
        print(f"No units for session {sess_name}, skipping")
        return None, False

    beh, donor, draws = load_beh(sess_name, args)
    if beh is None or not is_qualifying(beh, args):
        print(f"Dropping session {sess_name}, no qualifying behavior after {draws} donor draws")
        return None, True

    role_to_feat = assign_roles(beh, sess_name, args)
    feat_to_role = {feat: role for role, feat in role_to_feat.items()}
    beh = beh[(beh.PreferredBeliefProb > BELIEF_PARTITION_THRESH) & beh.PreferredBelief.isin(feat_to_role)].copy()
    beh["condition"] = beh.PreferredBelief.map(feat_to_role)
    beh = behavioral_utils.balance_trials_by_condition(beh, ["condition"])

    roles = pd.DataFrame({
        "session": sess_name,
        "role": list(role_to_feat),
        "dim": [args.dim_a if role.startswith("A") else args.dim_b for role in role_to_feat],
        "feature": list(role_to_feat.values()),
        "n_trials": len(beh) // len(ROLES),
        "donor_session": donor,
        "donor_draws": draws,
    })
    return {"session": sess_name, "beh": beh, "frs": frs, "roles": roles}, False


def dim_ccgp_heldout(args):
    file_name = belief_partitions_io.get_dim_ccgp_heldout_file_name(args)
    output_dir = belief_partitions_io.get_dir_name(args)

    loaded = [load_session(row, args) for _, row in args.sessions.iterrows()]
    sessions = [sess for sess, _ in loaded if sess is not None]
    n_dropped = sum(dropped for _, dropped in loaded)
    if len(sessions) == 0:
        raise ValueError("no sessions left to decode")
    unit_ids = np.sort(np.concatenate([sess["frs"].PseudoUnitID.unique() for sess in sessions]))
    print(f"{len(sessions)} sessions, {len(unit_ids)} units, {n_dropped} sessions dropped")

    classes = [args.dim_a, args.dim_b]
    label_map = {role: args.dim_a if role.startswith("A") else args.dim_b for role in ROLES}
    ti = args.trial_interval
    time_bins = np.arange(0, ti.pre_interval / 1000, ti.interval_size / 1000)
    # the held-out pair is never trained on, so all of its trials are test trials
    test_args = copy.copy(args)
    test_args.test_ratio = 1.0

    rows = []
    for fold in range(MIN_FEATS):
        held_out = [f"A{fold + 1}", f"B{fold + 1}"]
        train_roles = [role for role in ROLES if role not in held_out]
        print(f"Fold {fold}: train {train_roles}, test {held_out}", flush=True)
        train_datas = make_sess_datas(sessions, train_roles, args)
        train_accs, models = train_dim_decoder(train_datas, args, classes, label_map, time_bins)

        test_datas = make_sess_datas(sessions, held_out, test_args)
        ccgp_accs = pseudo_classifier_utils.evaluate_model_with_data(
            models, test_datas, time_bins,
            num_test_per_cond=args.num_test_per_cond, condition_label_map=label_map
        )
        if args.shuffle_idx is None:
            np.save(os.path.join(output_dir, f"{file_name}_fold_{fold}_models.npy"), models)

        for (time_idx, split_idx), ccgp_acc in np.ndenumerate(ccgp_accs):
            # bin end, as in belief_partitions_io.transform_np_acc_to_df
            time = (time_idx * ti.interval_size + ti.interval_size - ti.pre_interval) / 1000
            rows.append({
                "fold": fold, "held_out": "_".join(held_out),
                "split_idx": split_idx, "TimeIdx": int(round(time * 10)), "Time": time,
                "train_acc": train_accs[time_idx, split_idx], "ccgp_acc": ccgp_acc,
            })
    res = pd.DataFrame(rows)
    res.insert(0, "dim_pair_idx", args.dim_pair_idx)
    res.insert(1, "dim_a", args.dim_a)
    res.insert(2, "dim_b", args.dim_b)
    res["shuffle_idx"] = np.nan if args.shuffle_idx is None else args.shuffle_idx
    res["mode"] = MODE if args.shuffle_idx is None else f"{MODE}_shuffle"
    res["n_sessions"] = len(sessions)
    res["n_units"] = len(unit_ids)
    res["n_dropped_sessions"] = n_dropped
    res.to_pickle(os.path.join(output_dir, f"{file_name}.pickle"))

    pd.concat([sess["roles"] for sess in sessions], ignore_index=True).to_pickle(os.path.join(output_dir, f"{file_name}_roles.pickle"))
    if args.shuffle_idx is None:
        pd.DataFrame({"PseudoUnitID": unit_ids}).to_csv(os.path.join(output_dir, f"{file_name}_unit_ids.csv"))


def process_args(args):
    """
    Determines the dimension pair, its sessions, and the trial interval, adds them to args
    """
    if args.subject != "both":
        raise ValueError("dimension CCGP only supports combining subjects")
    dim_pairs = pd.read_pickle(DIM_PAIRS_PATH)
    row = dim_pairs.iloc[args.dim_pair_idx]
    args.dim_a = row.dim_a
    args.dim_b = row.dim_b

    valid_sess = pd.concat([pd.read_pickle(SESSIONS_PATH.format(sub=sub)) for sub in ["SA", "BL"]])
    args.sessions = valid_sess[valid_sess.session_name.isin(row.sessions)]
    # shuffle donors are drawn from here
    args.all_sessions = args.sessions
    args.trial_interval = get_trial_interval(args.trial_event)
    args.time_range = PRE_STIM_RANGE
    args.mode = MODE

    print(f"Held-out dimension CCGP for {args.dim_a} vs {args.dim_b} using {len(args.sessions)} sessions", flush=True)
    print(f"Shuffle {args.shuffle_idx}", flush=True)
    print(f"Using units {args.sig_unit_level}", flush=True)
    return args


def main():
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(BeliefPartitionConfigs(), parser)
    parser.add_argument("--dim_pair_idx", type=int, required=True)
    args = parser.parse_args()

    process_args(args)
    dim_ccgp_heldout(args)


if __name__ == "__main__":
    main()
