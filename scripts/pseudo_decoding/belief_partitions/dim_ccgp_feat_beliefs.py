"""
Dimension CCGP of feature beliefs, for one quadruple of features [X, X', Y, Y'] from
generate_dim_quads.py, where X, X' share dimension A and Y, Y' share dimension B. A decoder trained
on High X vs High Y, labelled by dimension, is tested on High X' vs High Y'. Design:
claude_notes/dimension_ccgp_design_brainstorm.md

All four train/test assignments are run, in this order (assignment_idx 0-3):

    (X, Y) -> (X', Y'),   (X, Y') -> (X', Y),   (X', Y) -> (X, Y'),   (X', Y') -> (X, Y)

High f means PreferredBelief == f and PreferredBeliefProb > BELIEF_PARTITION_THRESH. Decoding is
over the pre-stimulus bins, [-1, 0]s around StimOnset, on the union of the four features'
--sig_unit_level units. Each pair's trials are balanced and split as in ccgp_feat_beliefs.py, and
CCGP scores each split's test-portion trials of the test pair.

With --shuffle_idx set, behavior is session permuted, donors drawn from the quadruple's own
sessions. A donor truncated to the session's length can lack one of the four conditions; that
session is then dropped from the whole run, since its units must be present both to train and to
test, and counted in n_dropped_sessions.

Writes {stem}.pickle per run, one row per (assignment, split, bin), with the train pair's held-out
accuracy (train_pair_acc) and the CCGP accuracy (ccgp_acc). True runs also write
{stem}_assignment_{i}_models.npy per assignment (bins x splits) and {stem}_unit_ids.csv, the
PseudoUnitIDs in the decoder's input order.
"""

import os
import copy
import argparse
import numpy as np
import pandas as pd

import utils.behavioral_utils as behavioral_utils
import utils.spike_utils as spike_utils
import utils.pseudo_classifier_utils as pseudo_classifier_utils
import utils.session_data as session_data
from constants.behavioral_constants import *
from constants.decoding_constants import *

from models.trainer import Trainer
from models.model_wrapper import ModelWrapper
from models.multinomial_logistic_regressor import NormedDropoutMultinomialLogisticRegressor

from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import BeliefPartitionConfigs, add_defaults_to_parser
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
from scripts.pseudo_decoding.belief_partitions.decode_belief_partitions import DATA_MODE
from scripts.pseudo_decoding.belief_partitions.generate_dim_quads import QUADS_PATH

MODE = "dim_ccgp"
PRE_STIM_RANGE = [-1000, 0]
# ConditionTrialSplitter's minimum trials per condition
MIN_TRIALS_PER_COND = 2


def get_assignments(quad):
    """
    The four (train pair, test pair) assignments of quad = [X, X', Y, Y'], each pair (dim A feat, dim B feat)
    """
    x, x_p, y, y_p = quad
    return [
        ((x, y), (x_p, y_p)),
        ((x, y_p), (x_p, y)),
        ((x_p, y), (x, y_p)),
        ((x_p, y_p), (x, y)),
    ]


def load_session(row, args):
    """
    Loads a session's High trials of the four features, condition = preferred feature, and its
    firing rates. Returns (session dict or None, dropped), where dropped marks a session left out
    for lacking a condition, as opposed to having no units.
    """
    sess_name = row.session_name
    sub_args = copy.deepcopy(args)
    sub_args.subject = behavioral_utils.get_sub_for_session(sess_name)

    frs = spike_utils.get_frs_from_args(sub_args, sess_name)
    frs = frs.rename(columns={DATA_MODE: "Value"})
    if len(frs) == 0:
        print(f"No units for session {sess_name}, skipping")
        return None, False

    beh = behavioral_utils.load_behavior_from_args(sess_name, sub_args)
    beh = beh[(beh.PreferredBeliefProb > BELIEF_PARTITION_THRESH) & beh.PreferredBelief.isin(args.feat_quad)].copy()
    beh["condition"] = beh.PreferredBelief
    counts = beh.condition.value_counts().reindex(args.feat_quad, fill_value=0)
    if (counts < MIN_TRIALS_PER_COND).any():
        print(f"Dropping session {sess_name}, too few trials per condition: {counts.to_dict()}")
        return None, True
    return {"session": sess_name, "beh": beh, "frs": frs}, False


def make_sess_datas(sessions, pair, args):
    """
    Per session, balances the High trials of the pair's two features, splits them train/test
    """
    def make(sess):
        beh = sess["beh"][sess["beh"].condition.isin(pair)]
        beh = behavioral_utils.balance_trials_by_condition(beh, ["condition"])
        return session_data.create_from_splitter(args, "condition", sess["session"], beh, sess["frs"])
    return pd.Series([make(sess) for sess in sessions])


def train_dim_decoder(sess_datas, args, classes, label_map, time_bins):
    """
    Same as decode_belief_partitions.train_decoder, with the quadruple's dimensions as classes
    """
    num_neurons = sess_datas.apply(lambda x: x.get_num_neurons()).sum()
    print(f"using {num_neurons} neurons")
    init_params = {"n_inputs": num_neurons, "p_dropout": args.p_dropout, "n_classes": len(classes)}
    trainer = Trainer(learning_rate=args.learning_rate, max_iter=args.max_iter)
    model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, classes)
    _, test_accs, _, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(
        model, sess_datas, time_bins,
        args.num_splits, args.num_train_per_cond, args.num_test_per_cond,
        condition_label_map=label_map
    )
    return test_accs, models


def dim_ccgp(args):
    file_name = belief_partitions_io.get_dim_ccgp_file_name(args)
    output_dir = belief_partitions_io.get_dir_name(args)

    loaded = [load_session(row, args) for _, row in args.sessions.iterrows()]
    sessions = [sess for sess, _ in loaded if sess is not None]
    n_dropped = sum(dropped for _, dropped in loaded)
    if len(sessions) == 0:
        raise ValueError("no sessions left to decode")
    unit_ids = np.sort(np.concatenate([sess["frs"].PseudoUnitID.unique() for sess in sessions]))
    print(f"{len(sessions)} sessions, {len(unit_ids)} units, {n_dropped} sessions dropped")

    x, x_p, y, y_p = args.feat_quad
    classes = [args.dim_a, args.dim_b]
    label_map = {x: args.dim_a, x_p: args.dim_a, y: args.dim_b, y_p: args.dim_b}
    ti = args.trial_interval
    time_bins = np.arange(0, ti.pre_interval / 1000, ti.interval_size / 1000)

    rows = []
    for assignment_idx, (train_pair, test_pair) in enumerate(get_assignments(args.feat_quad)):
        print(f"Assignment {assignment_idx}: train {train_pair}, test {test_pair}", flush=True)
        train_datas = make_sess_datas(sessions, train_pair, args)
        train_pair_accs, models = train_dim_decoder(train_datas, args, classes, label_map, time_bins)

        test_datas = make_sess_datas(sessions, test_pair, args)
        ccgp_accs = pseudo_classifier_utils.evaluate_model_with_data(
            models, test_datas, time_bins,
            num_test_per_cond=args.num_test_per_cond, condition_label_map=label_map
        )
        if args.shuffle_idx is None:
            np.save(os.path.join(output_dir, f"{file_name}_assignment_{assignment_idx}_models.npy"), models)

        for (time_idx, split_idx), ccgp_acc in np.ndenumerate(ccgp_accs):
            # bin end, as in belief_partitions_io.transform_np_acc_to_df
            time = (time_idx * ti.interval_size + ti.interval_size - ti.pre_interval) / 1000
            rows.append({
                "assignment_idx": assignment_idx,
                "assignment": f"{train_pair[0]}_{train_pair[1]}→{test_pair[0]}_{test_pair[1]}",
                "train_a": train_pair[0], "train_b": train_pair[1],
                "test_a": test_pair[0], "test_b": test_pair[1],
                "split_idx": split_idx, "TimeIdx": int(round(time * 10)), "Time": time,
                "train_pair_acc": train_pair_accs[time_idx, split_idx], "ccgp_acc": ccgp_acc,
            })
    res = pd.DataFrame(rows)
    res.insert(0, "quad_idx", args.quad_idx)
    res.insert(1, "quad_str", "_".join(args.feat_quad))
    res.insert(2, "dim_a", args.dim_a)
    res.insert(3, "dim_b", args.dim_b)
    res["shuffle_idx"] = np.nan if args.shuffle_idx is None else args.shuffle_idx
    res["mode"] = MODE if args.shuffle_idx is None else f"{MODE}_shuffle"
    res["n_sessions"] = len(sessions)
    res["n_units"] = len(unit_ids)
    res["n_dropped_sessions"] = n_dropped
    res.to_pickle(os.path.join(output_dir, f"{file_name}.pickle"))

    if args.shuffle_idx is None:
        pd.DataFrame({"PseudoUnitID": unit_ids}).to_csv(os.path.join(output_dir, f"{file_name}_unit_ids.csv"))


def process_args(args):
    """
    Determines the quadruple, its sessions, and the trial interval, adds them to args
    """
    if args.subject != "both":
        raise ValueError("dimension CCGP only supports combining subjects")
    quads = pd.read_pickle(QUADS_PATH)
    row = quads.iloc[args.quad_idx]
    args.feat_quad = list(row.quad)
    args.dim_a = row.dim_a
    args.dim_b = row.dim_b

    valid_sess = pd.concat([pd.read_pickle(SESSIONS_PATH.format(sub=sub)) for sub in ["SA", "BL"]])
    args.sessions = valid_sess[valid_sess.session_name.isin(row.sessions)]
    # session permutation draws its donors from here
    args.all_sessions = args.sessions
    args.trial_interval = get_trial_interval(args.trial_event)
    args.time_range = PRE_STIM_RANGE
    args.mode = MODE

    print(f"Dimension CCGP for quad {args.feat_quad} ({args.dim_a} vs {args.dim_b}) using {len(args.sessions)} sessions", flush=True)
    print(f"Shuffle {args.shuffle_idx}, method {args.shuffle_method}", flush=True)
    print(f"Using units {args.sig_unit_level}", flush=True)
    return args


def main():
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(BeliefPartitionConfigs(), parser)
    parser.add_argument("--quad_idx", type=int, required=True)
    args = parser.parse_args()

    process_args(args)
    dim_ccgp(args)


if __name__ == "__main__":
    main()
