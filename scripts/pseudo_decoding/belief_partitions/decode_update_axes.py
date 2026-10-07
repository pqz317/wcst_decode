"""
Pref and conf decoders for decoder_update_projections.py, one per pair split of
mean_diff_update_projections, trained only on that split's axis trials.

For each (session, feature X) and each of --num_splits splits, the pair split draws half of each
chose-X cell (outcome x partition) as test trials k and reserves k and k + 1; every other trial is
an axis trial. Split i's decoder trains on split i's axis trials, balanced by class, and its test
accuracy is scored on split i's test trials k. So no trial a split's projections score was trained
on by that split's decoder. The splits are rebuilt from behavior alone, so
decoder_update_projections.py reproduces them; the splits table is saved to check that.

Otherwise as the old no-cond decoders of decode_belief_partitions.py: no behavior filters, units
from --sig_unit_level, the same decoder. Only the 10 pre-stimulus bins are trained, which are the
first 10 bins of the StimOnset interval, so belief_partitions_io.read_models gives them the same
Time values as the old models. Model column i of the saved array is split i.

--split_method chunk replaces the pair split with a chunk split: each session's valid trials are
cut into repeating cycles of an axis block followed by a --chunk_test_len trial test block, the
axis block sized so axis trials are about --chunk_train_frac of the cycle, from a random starting
point per split. Every trial of an axis block is an axis trial. A chose-X trial k in a test block
whose next trial is in the same block is a test trial; the rest of the test block is reserved.
Which trials are excluded from training then depends only on their position in the session, not on
behavior, so the axis trials keep the session's natural trial history.

Launched by slurm_launch_decode_update_axes.sh, one job per (mode, feature).
"""

import os
import copy
import argparse
import numpy as np
import pandas as pd

import utils.pseudo_classifier_utils as pseudo_classifier_utils
import utils.behavioral_utils as behavioral_utils
import utils.spike_utils as spike_utils
import utils.session_data as session_data
from constants.behavioral_constants import *
from constants.decoding_constants import *
from models.trainer import Trainer
from models.model_wrapper import ModelWrapper
from models.multinomial_logistic_regressor import NormedDropoutMultinomialLogisticRegressor
from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import BeliefPartitionConfigs, add_defaults_to_parser
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
from scripts.pseudo_decoding.belief_partitions.decode_belief_partitions import find_valid_sessions_for_feat_sub
from scripts.pseudo_decoding.belief_partitions.prior_dependent_updates import load_beh, PRE_STIM_RANGE
import scripts.pseudo_decoding.belief_partitions.mean_diff_update_projections as md

OUTPUT_PATH = "/data/patrick_res/update_axes_decoders"
DEFAULT_TEST_FRAC = 0.5
SPLIT_METHODS = ["pair", "chunk"]
DEFAULT_CHUNK_TEST_LEN = 20
# PROVISIONAL: 0.3 is the training share the 10- vs 20-trial block comparison used
DEFAULT_CHUNK_TRAIN_FRAC = 0.3

# TimeBins of the pre-stimulus bins, in seconds from the start of the StimOnset interval
PRE_STIM_BINS = np.arange(0, 1.0, 0.1)


def decoder_base_path(test_frac, split_method="pair", chunk_test_len=DEFAULT_CHUNK_TEST_LEN,
                      chunk_train_frac=DEFAULT_CHUNK_TRAIN_FRAC):
    """
    Where decoders for a split setting live, so runs with different settings don't overwrite each
    other. Pair split: OUTPUT_PATH at the default --test_frac, a test_frac_{f} subfolder otherwise.
    Chunk split: a chunk_{test len}_train_frac_{f} subfolder.
    """
    if split_method == "chunk":
        return os.path.join(OUTPUT_PATH, f"chunk_{chunk_test_len}_train_frac_{chunk_train_frac:g}")
    if np.isclose(test_frac, DEFAULT_TEST_FRAC):
        return OUTPUT_PATH
    return os.path.join(OUTPUT_PATH, f"test_frac_{test_frac:g}")


def args_base_path(args):
    """
    decoder_base_path for the split settings in args.
    """
    return decoder_base_path(args.test_frac, args.split_method, args.chunk_test_len, args.chunk_train_frac)


def chunk_split(sess_name, trials, seed, split_idx, test_len, train_frac):
    """
    Adds `half` to label_trials output by position in the session's valid-trial sequence: cycles
    of round(test_len * train_frac / (1 - train_frac)) axis trials, then test_len test-block
    trials. In a test block, a chose-X trial whose next valid trial is in the same block is "test";
    every other test-block trial is "reserved". Axis blocks are "axis". Also adds `in_block_pair`,
    true for every test-block trial, chose X or not, whose next valid trial is in the same block, so
    decoder_update_projections.py can project the not-X ones too.

    The starting point within the cycle is drawn per (session, seed, split_idx).
    PROVISIONAL: it doesn't depend on the feature, so within a split every feature shares the same
    blocks.
    """
    axis_len = max(1, int(round(test_len * train_frac / (1 - train_frac))))
    cycle = axis_len + test_len
    phase = np.random.default_rng([int(sess_name), seed, split_idx]).integers(cycle)
    trials = trials.sort_values("TrialNumber").reset_index(drop=True)
    in_test_block = ((np.arange(len(trials)) + phase) % cycle) >= axis_len
    # the next row is the next valid trial; it must also be in this test block, and not across a
    # block edge
    next_in_block = np.append(in_test_block[1:] & (((np.arange(1, len(trials)) + phase) % cycle) != axis_len), False)
    next_is_next = np.append(trials.TrialNumber.to_numpy()[1:] == trials.NextTrialNumber.to_numpy()[:-1], False)
    trials["in_block_pair"] = in_test_block & next_in_block & next_is_next
    is_test = trials.in_block_pair.to_numpy() & trials.chose.to_numpy()
    trials["half"] = np.where(~in_test_block, "axis", np.where(is_test, "test", "reserved"))
    # no trial of any test pair is an axis trial
    test = trials[trials.half == "test"]
    axis_trials = set(trials.loc[trials.half == "axis", "TrialNumber"])
    assert axis_trials.isdisjoint(set(test.TrialNumber) | set(test.NextTrialNumber))
    return trials


def pair_splits(sess_name, feat, args):
    """
    {split_idx: labeled trials with `half`} for the session's --num_splits splits: pair splits
    drawing --test_frac of every chose-X cell as test trials, or with --split_method chunk, chunk
    splits.
    """
    labeled = md.label_trials(load_beh(sess_name, args), feat)
    if args.split_method == "chunk":
        return {
            i: chunk_split(sess_name, labeled.copy(), args.train_test_seed, i, args.chunk_test_len, args.chunk_train_frac)
            for i in range(args.num_splits)
        }
    return {
        i: md.split_pairs(sess_name, feat, labeled.copy(), args.train_test_seed, split_idx=i, test_frac=args.test_frac)
        for i in range(args.num_splits)
    }


class EmptyClassError(ValueError):
    """
    A (session, split) with no train or no test trials for some class.
    """


def splits_table(sess_name, beh, splits):
    """
    Train/test trials per (split, class), in ConditionTrialSplitter's format: train is the split's
    axis trials, balanced by class; test is the split's test trials k. beh is labeled by mode, so
    trials outside the mode's classes (Low, for pref) are already dropped.

    Raises if any class has no train or no test trials, since pseudo-trials can't be drawn from an
    empty set.
    """
    rows = []
    for i, trials in splits.items():
        half = beh.TrialNumber.map(trials.set_index("TrialNumber").half)
        train = behavioral_utils.balance_trials_by_condition(beh[half == "axis"], condition_columns=["condition"])
        test = beh[half == "test"]
        for cond in sorted(beh.condition.unique()):
            row = {
                "Condition": cond,
                "TrainTrials": train[train.condition == cond].TrialNumber.to_numpy(),
                "TestTrials": test[test.condition == cond].TrialNumber.to_numpy(),
                "split_idx": i, "session": sess_name,
            }
            if len(row["TrainTrials"]) == 0 or len(row["TestTrials"]) == 0:
                raise EmptyClassError(f"session {sess_name} split {i} class {cond}: "
                                 f"{len(row['TrainTrials'])} train, {len(row['TestTrials'])} test trials")
            rows.append(row)
    return pd.DataFrame(rows)


def load_session_data(sess_name, args):
    """
    SessionData whose splits are the session's pair or chunk splits, or None if it has no firing
    rates, or if any split leaves a class with no train or test trials.
    As decode_belief_partitions.load_session_data with no filters.
    """
    beh = behavioral_utils.load_behavior_from_args(sess_name, args)
    beh = behavioral_utils.get_feat_choice_label(beh, args.feat)
    beh = behavioral_utils.get_belief_partitions(beh, args.feat, use_x=True)
    beh = behavioral_utils.get_label_by_mode(beh, args.mode)
    try:
        splits_df = splits_table(sess_name, beh, pair_splits(sess_name, args.feat, args))
    except EmptyClassError as e:
        # HACK: the session is left out of this feature's decoders in every split, not just the one
        # with the empty class, because evaluate_classifiers_by_time_bins needs the same units in
        # all splits. Its units then get no axis for this feature, so decoder_update_projections.py
        # drops this (session, feature) from the stats. Known case: the chunk split at the defaults
        # leaves session 20181005 with no High X training trials for GREEN in splits 5 and 6, since
        # all 27 of its High X trials are close together and fall inside test blocks.
        print(f"HACK: dropping session {sess_name} from {args.feat} {args.mode}: {e}", flush=True)
        return None

    frs = spike_utils.get_frs_from_args(args, sess_name)
    frs = frs.rename(columns={"FiringRate": "Value"})
    if len(frs) == 0:
        return None
    return session_data.SessionData(sess_name, beh, frs, splits_df)


def load_session_datas(args):
    datas = []
    for sub in ["SA", "BL"]:
        sub_args = copy.deepcopy(args)
        sub_args.subject = sub
        for sess_name in find_valid_sessions_for_feat_sub(sub_args).session_name:
            datas.append(load_session_data(sess_name, sub_args))
    return pd.Series([d for d in datas if d is not None])


def add_split_args(parser):
    """
    The chunk split arguments, shared with decoder_update_projections.py, which must be given the
    same values to read these decoders and rebuild their splits. --test_frac applies to the pair
    split only.
    """
    parser.add_argument("--split_method", default="pair", choices=SPLIT_METHODS)
    parser.add_argument("--chunk_test_len", default=DEFAULT_CHUNK_TEST_LEN, type=int)
    parser.add_argument("--chunk_train_frac", default=DEFAULT_CHUNK_TRAIN_FRAC, type=float)
    return parser


def train_decoder(sess_datas, args):
    """
    As decode_belief_partitions.train_decoder, over the pre-stimulus bins only.
    """
    classes = MODE_TO_CLASSES[args.mode]
    num_neurons = sess_datas.apply(lambda x: x.get_num_neurons()).sum()
    print(f"using {num_neurons} neurons", flush=True)
    init_params = {"n_inputs": num_neurons, "p_dropout": args.p_dropout, "n_classes": len(classes)}
    trainer = Trainer(learning_rate=args.learning_rate, max_iter=args.max_iter)
    model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, classes)
    _, test_accs, _, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(
        model, sess_datas, PRE_STIM_BINS,
        args.num_splits, args.num_train_per_cond, args.num_test_per_cond,
        condition_label_map=MODE_COND_LABEL_MAPS[args.mode]
    )
    return test_accs, models


def main(args):
    args.feat = FEATURES[args.feat_idx]
    args.trial_interval = get_trial_interval(args.trial_event)
    args.time_range = PRE_STIM_RANGE
    args.base_output_path = args_base_path(args)
    split_str = (f"chunk {args.chunk_test_len}, train_frac {args.chunk_train_frac}" if args.split_method == "chunk"
                 else f"pair, test_frac {args.test_frac}")
    print(f"feat {args.feat}, mode {args.mode}, units {args.sig_unit_level}, split {split_str}", flush=True)

    sess_datas = load_session_datas(args)
    test_accs, models = train_decoder(sess_datas, args)

    output_dir = belief_partitions_io.get_dir_name(args)
    file_name = belief_partitions_io.get_file_name(args)
    np.save(os.path.join(output_dir, f"{file_name}_test_accs.npy"), test_accs)
    np.save(os.path.join(output_dir, f"{file_name}_models.npy"), models)
    pd.concat(sess_datas.apply(lambda x: x.get_splits_df()).values).to_pickle(os.path.join(output_dir, f"{file_name}_splits.pickle"))
    unit_ids = pd.DataFrame({"PseudoUnitIDs": np.concatenate(sess_datas.apply(lambda x: x.get_pseudo_unit_ids()).values)})
    unit_ids.to_csv(os.path.join(output_dir, f"{file_name}_unit_ids.csv"))
    print(f"saved {file_name} to {output_dir}; pre-stim test acc {test_accs.mean():.3f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(
        BeliefPartitionConfigs(subject="both", base_output_path=OUTPUT_PATH), parser
    )
    # fraction of each chose-X cell drawn as test trials k (k and k + 1 then reserved) per split
    parser.add_argument("--test_frac", default=DEFAULT_TEST_FRAC, type=float)
    add_split_args(parser)
    main(parser.parse_args())
