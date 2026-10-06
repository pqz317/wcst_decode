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

# TimeBins of the pre-stimulus bins, in seconds from the start of the StimOnset interval
PRE_STIM_BINS = np.arange(0, 1.0, 0.1)


def pair_splits(sess_name, feat, args):
    """
    {split_idx: split_pairs output} for the session's --num_splits pair splits.
    """
    labeled = md.label_trials(load_beh(sess_name, args), feat)
    return {
        i: md.split_pairs(sess_name, feat, labeled.copy(), args.train_test_seed, split_idx=i)
        for i in range(args.num_splits)
    }


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
                raise ValueError(f"session {sess_name} split {i} class {cond}: "
                                 f"{len(row['TrainTrials'])} train, {len(row['TestTrials'])} test trials")
            rows.append(row)
    return pd.DataFrame(rows)


def load_session_data(sess_name, args):
    """
    SessionData whose splits are the session's pair splits, or None if it has no firing rates.
    As decode_belief_partitions.load_session_data with no filters.
    """
    beh = behavioral_utils.load_behavior_from_args(sess_name, args)
    beh = behavioral_utils.get_feat_choice_label(beh, args.feat)
    beh = behavioral_utils.get_belief_partitions(beh, args.feat, use_x=True)
    beh = behavioral_utils.get_label_by_mode(beh, args.mode)
    splits_df = splits_table(sess_name, beh, pair_splits(sess_name, args.feat, args))

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
    print(f"feat {args.feat}, mode {args.mode}, units {args.sig_unit_level}", flush=True)

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
    main(parser.parse_args())
