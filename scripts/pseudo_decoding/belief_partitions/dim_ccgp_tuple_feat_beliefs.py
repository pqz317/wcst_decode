"""
Dimension CCGP of feature beliefs on a fixed feature 6-tuple from generate_dim_tuples.py: three
features of dimension A and three of dimension B, the same in every session. Proposal:
claude_notes/dimension_ccgp_fixed_tuples_proposal.md

The tuple's features are assigned at random to the roles A1-A3, B1-B3, seeded by tuple only, so the
true run and every shuffle share the assignment. Each session's High trials (PreferredBelief == f,
PreferredBeliefProb > BELIEF_PARTITION_THRESH) passing --beh_filters are balanced to the minimum
count over the six features. Decoding is over every bin of --trial_event's interval.

Fold k (0-2) holds out (A_k, B_k) and trains on the other two roles of each dimension, in role order
(A_i, A_j, B_i, B_j), i < j. Two decoders are trained per fold on the same trials and splits:
    dimension: (A_i, A_j) vs (B_i, B_j), tested on A_k vs B_k, labelled by dimension
    random:    (A_i, B_j) vs (B_i, A_j), the second-slot features swapped, tested on A_k vs B_k,
               A_k taking A_i's class (mix_0) and B_k B_i's (mix_1)
so a random fold's training axis has no net dimension component.

With --shuffle_idx set, each session's behavior is replaced by a donor's from the tuple's sessions,
truncated to the session's length. A donor in which some feature lacks >= BLOCK_THRESH blocks or
>= MIN_TRIALS_PER_COND High trials is redrawn, up to MAX_DONOR_DRAWS donors, after which the session
is dropped from that run and counted in n_dropped_sessions.

Units are --sig_unit_level's units for any of the tuple's features (spike_utils.get_sig_units).

Writes per run {stem}.pickle, one row per (fold_type, fold, split, bin), with the held-out accuracy
on the training features (train_acc) and the accuracy on the held-out pair (ccgp_acc);
{stem}_roles.pickle, each kept session's role -> feature mapping, balanced trial count and donor;
and {stem}_folds.pickle, each (fold_type, fold)'s feature -> class map. True runs also write
{stem}_{fold_type}_fold_{k}_models.npy (bins x splits) and {stem}_unit_ids.csv, the PseudoUnitIDs
in the decoder's input order.
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
    DATA_MODE, MIN_TRIALS_PER_COND, make_sess_datas, train_dim_decoder
)
from scripts.pseudo_decoding.belief_partitions.generate_dim_tuples import TUPLES_PATH, BLOCK_THRESH, FEATS_PER_DIM

MODE = "dim_ccgp_tuples_full"
ROLES = [f"{prefix}{i + 1}" for prefix in ["A", "B"] for i in range(FEATS_PER_DIM)]
FOLD_TYPES = ["dimension", "random"]
MAX_DONOR_DRAWS = 10


def assign_roles(args):
    """
    The tuple's features assigned at random to roles, seeded by tuple only, returns {role: feature}
    """
    rng = np.random.default_rng([args.train_test_seed, args.tuple_idx])
    role_to_feat = {}
    for prefix, feats in [("A", args.feats_a), ("B", args.feats_b)]:
        for i, feat in enumerate(rng.permutation(feats)):
            role_to_feat[f"{prefix}{i + 1}"] = feat
    return role_to_feat


def get_folds(role_to_feat, dim_a, dim_b):
    """
    Per fold k, the held-out features (A_k, B_k), the training features (A_i, A_j, B_i, B_j) with
    i < j, and per fold type its classes and feature -> class map, covering train and test features
    """
    folds = []
    for k in range(FEATS_PER_DIM):
        i, j = [idx for idx in range(FEATS_PER_DIM) if idx != k]
        a_i, a_j, a_k = [role_to_feat[f"A{idx + 1}"] for idx in [i, j, k]]
        b_i, b_j, b_k = [role_to_feat[f"B{idx + 1}"] for idx in [i, j, k]]
        fold_types = {
            "dimension": ([dim_a, dim_b], {a_i: dim_a, a_j: dim_a, a_k: dim_a, b_i: dim_b, b_j: dim_b, b_k: dim_b}),
            # the second-slot training features swapped between classes, all else kept
            "random": (["mix_0", "mix_1"], {a_i: "mix_0", b_j: "mix_0", a_k: "mix_0", b_i: "mix_1", a_j: "mix_1", b_k: "mix_1"}),
        }
        folds.append({"fold": k, "held_out": [a_k, b_k], "train": [a_i, a_j, b_i, b_j], "fold_types": fold_types})
    return folds


def is_qualifying(beh, args):
    """
    Whether every feature of the tuple was the rule for >= BLOCK_THRESH blocks and has
    >= MIN_TRIALS_PER_COND High trials passing beh_filters
    """
    num_blocks = beh.groupby("CurrentRule").BlockNumber.nunique()
    filt_beh = behavioral_utils.filter_behavior(beh, args.beh_filters)
    num_high = filt_beh[filt_beh.PreferredBeliefProb > BELIEF_PARTITION_THRESH].PreferredBelief.value_counts()
    return all(num_blocks.get(f, 0) >= BLOCK_THRESH and num_high.get(f, 0) >= MIN_TRIALS_PER_COND for f in args.feat_tuple)


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


def load_session(row, args):
    """
    Loads a session's balanced High trials of the tuple's six features, condition = feature, and its
    firing rates. Returns (session dict or None, dropped), where dropped marks a session left out
    for lacking qualifying behavior, as opposed to having no units.
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

    beh = behavioral_utils.filter_behavior(beh, args.beh_filters)
    beh = beh[(beh.PreferredBeliefProb > BELIEF_PARTITION_THRESH) & beh.PreferredBelief.isin(args.feat_tuple)].copy()
    beh["condition"] = beh.PreferredBelief
    beh = behavioral_utils.balance_trials_by_condition(beh, ["condition"])

    feat_to_role = {feat: role for role, feat in args.role_to_feat.items()}
    roles = pd.DataFrame({
        "session": sess_name,
        "role": [feat_to_role[f] for f in args.feat_tuple],
        "dim": [FEATURE_TO_DIM[f] for f in args.feat_tuple],
        "feature": args.feat_tuple,
        "n_trials": len(beh) // len(args.feat_tuple),
        "donor_session": donor,
        "donor_draws": draws,
    })
    return {"session": sess_name, "beh": beh, "frs": frs, "roles": roles}, False


def get_unit_ids(sessions):
    return np.sort(np.concatenate([sess["frs"].PseudoUnitID.unique() for sess in sessions]))


def dim_ccgp_tuple(args):
    file_name = belief_partitions_io.get_dim_ccgp_tuple_file_name(args)
    output_dir = belief_partitions_io.get_dir_name(args)

    loaded = [load_session(row, args) for _, row in args.sessions.iterrows()]
    sessions = [sess for sess, _ in loaded if sess is not None]
    n_dropped = sum(dropped for _, dropped in loaded)
    if len(sessions) == 0:
        raise ValueError("no sessions left to decode")
    unit_ids = get_unit_ids(sessions)
    print(f"{len(sessions)} sessions, {len(unit_ids)} units, {n_dropped} sessions dropped")

    ti = args.trial_interval
    time_bins = np.arange(0, (ti.pre_interval + ti.post_interval) / 1000, ti.interval_size / 1000)
    # the held-out pair is never trained on, so all of its trials are test trials
    test_args = copy.copy(args)
    test_args.test_ratio = 1.0

    rows, fold_maps = [], []
    for fold in get_folds(args.role_to_feat, args.dim_a, args.dim_b):
        k = fold["fold"]
        print(f"Fold {k}: train {fold['train']}, test {fold['held_out']}", flush=True)
        # built once, so both fold types train and test on the same trials and splits
        train_datas = make_sess_datas(sessions, fold["train"], args)
        test_datas = make_sess_datas(sessions, fold["held_out"], test_args)
        for fold_type in FOLD_TYPES:
            classes, label_map = fold["fold_types"][fold_type]
            train_classes = " vs ".join(
                "_".join(f for f in fold["train"] if label_map[f] == c) for c in classes
            )
            print(f"  {fold_type}: {train_classes}", flush=True)
            fold_maps.append(pd.DataFrame({
                "fold_type": fold_type, "fold": k, "feature": list(label_map), "class": list(label_map.values()),
                "held_out": [f in fold["held_out"] for f in label_map],
            }))
            train_accs, models = train_dim_decoder(train_datas, args, classes, label_map, time_bins)
            ccgp_accs = pseudo_classifier_utils.evaluate_model_with_data(
                models, test_datas, time_bins,
                num_test_per_cond=args.num_test_per_cond, condition_label_map=label_map
            )
            if args.shuffle_idx is None:
                np.save(os.path.join(output_dir, f"{file_name}_{fold_type}_fold_{k}_models.npy"), models)

            for (time_idx, split_idx), ccgp_acc in np.ndenumerate(ccgp_accs):
                # bin end, as in belief_partitions_io.transform_np_acc_to_df
                time = (time_idx * ti.interval_size + ti.interval_size - ti.pre_interval) / 1000
                rows.append({
                    "fold_type": fold_type, "fold": k, "held_out": "_".join(fold["held_out"]),
                    "train_classes": train_classes,
                    "split_idx": split_idx, "TimeIdx": int(round(time * 10)), "Time": time,
                    "train_acc": train_accs[time_idx, split_idx], "ccgp_acc": ccgp_acc,
                })
    res = pd.DataFrame(rows)
    res.insert(0, "tuple_idx", args.tuple_idx)
    res.insert(1, "dim_a", args.dim_a)
    res.insert(2, "dim_b", args.dim_b)
    res["n_sessions"] = len(sessions)
    res["n_units"] = len(unit_ids)
    res["shuffle_idx"] = np.nan if args.shuffle_idx is None else args.shuffle_idx
    res["mode"] = args.mode if args.shuffle_idx is None else f"{args.mode}_shuffle"
    res["n_dropped_sessions"] = n_dropped
    res.to_pickle(os.path.join(output_dir, f"{file_name}.pickle"))

    pd.concat([sess["roles"] for sess in sessions], ignore_index=True).to_pickle(os.path.join(output_dir, f"{file_name}_roles.pickle"))
    pd.concat(fold_maps, ignore_index=True).to_pickle(os.path.join(output_dir, f"{file_name}_folds.pickle"))
    if args.shuffle_idx is None:
        pd.DataFrame({"PseudoUnitID": unit_ids}).to_csv(os.path.join(output_dir, f"{file_name}_unit_ids.csv"))


def process_args(args):
    """
    Determines the tuple, its sessions, roles, and the trial interval, adds them to args
    """
    if args.subject != "both":
        raise ValueError("dimension CCGP only supports combining subjects")
    row = pd.read_pickle(TUPLES_PATH).iloc[args.tuple_idx]
    args.dim_a = row.dim_a
    args.dim_b = row.dim_b
    args.feats_a = list(row.feats_a)
    args.feats_b = list(row.feats_b)
    args.feat_tuple = list(row.feat_tuple)

    valid_sess = pd.concat([pd.read_pickle(SESSIONS_PATH.format(sub=sub)) for sub in ["SA", "BL"]])
    args.sessions = valid_sess[valid_sess.session_name.isin(row.sessions)]
    # shuffle donors are drawn from here
    args.all_sessions = args.sessions
    args.trial_interval = get_trial_interval(args.trial_event)
    # every bin of the interval
    args.time_range = None
    args.mode = MODE
    args.role_to_feat = assign_roles(args)

    print(f"Fixed-tuple dimension CCGP for tuple {args.tuple_idx}, {args.dim_a} vs {args.dim_b}, "
          f"roles {args.role_to_feat}, using {len(args.sessions)} sessions", flush=True)
    print(f"Shuffle {args.shuffle_idx}, event {args.trial_event}, filters {args.beh_filters}", flush=True)
    print(f"Using units {args.sig_unit_level}", flush=True)
    return args


def main():
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(BeliefPartitionConfigs(), parser)
    parser.add_argument("--tuple_idx", type=int, required=True)
    args = parser.parse_args()

    process_args(args)
    dim_ccgp_tuple(args)


if __name__ == "__main__":
    main()
