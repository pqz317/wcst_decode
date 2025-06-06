"""
Script for decoding various splits of the belief partition space
Do so a feature at a time
"""

import os
import numpy as np
import pandas as pd
import utils.pseudo_classifier_utils as pseudo_classifier_utils
import utils.behavioral_utils as behavioral_utils
import utils.spike_utils as spike_utils

from constants.behavioral_constants import *
from constants.decoding_constants import *
import utils.session_data as session_data

from models.trainer import Trainer
from models.model_wrapper import ModelWrapper
from models.multinomial_logistic_regressor import NormedDropoutMultinomialLogisticRegressor

import argparse
from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import BeliefPartitionConfigs, add_defaults_to_parser
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
import copy

FEATS_PATH = "/data/patrick_res/sessions/{sub}/feats_at_least_3blocks.pickle"
SESSIONS_PATH = "/data/patrick_res/sessions/{sub}/valid_sessions.pickle"
UNITS_PATH = "/data/patrick_res/firing_rates/{sub}/all_units.pickle"
SESS_BEHAVIOR_PATH = "/data/patrick_res/behavior/SA/{sess_name}_object_features.csv"

DATA_MODE = "FiringRate"

MODE_TO_CLASSES = {
    "conf": ["Low", "High"],
    "pref": ["High X", "High Not X"],
    "feat_belief": ["Low", "High X"],
    "policy": ["X", "Not X"]
}

def load_session_data(row, args, splits_df=None):
    """
    Loads behavior, neural data, prepares them for decoding
    """
    sess_name = row.session_name
    
    beh = behavioral_utils.load_behavior_from_args(sess_name, args)
    beh = behavioral_utils.get_feat_choice_label(beh, args.feat)
    if args.balance_by_filters: 
        beh = behavioral_utils.balance_trials_by_condition(beh, list(args.beh_filters.keys()))
    beh = behavioral_utils.filter_behavior(beh, args.beh_filters)
    beh = behavioral_utils.get_belief_partitions_by_mode(beh, args.feat, args.mode)
    beh = behavioral_utils.balance_trials_by_condition(beh, ["PartitionLabel"])

    frs = spike_utils.get_frs_from_args(args, sess_name)
    frs = frs.rename(columns={DATA_MODE: "Value"})
    if len(frs) == 0 or len(beh) == 0:
        return None
    
    if splits_df is None: 
        sess_data = session_data.create_from_splitter(args, "PartitionLabel", sess_name, beh, frs)
    else: 
        sess_data = session_data.create_from_splits_df(sess_name, beh, frs, splits_df)
    return sess_data

def train_decoder(sess_datas, args):
    # train the network
    # setup decoder, specify all possible label classes, number of neurons, parameters
    classes = MODE_TO_CLASSES[args.mode]
    num_neurons = sess_datas.apply(lambda x: x.get_num_neurons()).sum()
    init_params = {"n_inputs": num_neurons, "p_dropout": args.p_dropout, "n_classes": len(classes)}
    # create a trainer object
    trainer = Trainer(learning_rate=args.learning_rate, max_iter=args.max_iter)
    # create a wrapper for the decoder
    model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, classes)

    # calculate time bins (in seconds)
    trial_interval = args.trial_interval
    time_bins = np.arange(0, (trial_interval.post_interval + trial_interval.pre_interval) / 1000, trial_interval.interval_size / 1000)
    _, test_accs, _, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(
        model, sess_datas, time_bins, args.num_splits, args.num_train_per_cond, args.num_test_per_cond
    )
    return test_accs, models

def load_session_datas_for_sub(args, splits_df=None):
    print(f"reading data from {len(args.sub_sessions)} sessions for {args.subject}")
    sess_datas = args.sub_sessions.apply(lambda row: load_session_data(
        row, args, splits_df
    ), axis=1)
    sess_datas = sess_datas.dropna()
    return sess_datas

def find_valid_sessions_for_feat_sub(args):
    feat_sessions = pd.read_pickle(FEATS_PATH.format(sub=args.subject))
    valid_sess = pd.read_pickle(SESSIONS_PATH.format(sub=args.subject))
    row = feat_sessions[feat_sessions.feat == args.feat].iloc[0]
    sessions = valid_sess[valid_sess.session_name.isin(row.sessions)]
    return sessions

def load_session_datas(args, splits_df=None):
    if args.subject == "both":
        sa_args = copy.deepcopy(args)
        sa_args.subject = "SA"
    
        bl_args = copy.deepcopy(args)
        bl_args.subject = "BL"

        sa_sessions = find_valid_sessions_for_feat_sub(sa_args)
        bl_sessions = find_valid_sessions_for_feat_sub(bl_args)

        all_sessions = pd.concat((sa_sessions, bl_sessions))

        sa_args.sub_sessions = sa_sessions
        sa_args.all_sessions = all_sessions
        sa_datas = load_session_datas_for_sub(sa_args, splits_df)

        bl_args.sub_sessions = bl_sessions
        bl_args.all_sessions = all_sessions
        bl_datas = load_session_datas_for_sub(bl_args, splits_df)

        return pd.concat((sa_datas, bl_datas))
    else: 
        sessions = find_valid_sessions_for_feat_sub(args)
        args.sub_sessions = sessions
        args.all_sessions = sessions
        return load_session_datas_for_sub(args, splits_df)


def decode(args):
    sess_datas = load_session_datas(args)
    # naming for files, directory
    file_name = belief_partitions_io.get_file_name(args)
    output_dir = belief_partitions_io.get_dir_name(args)

    test_accs, models = train_decoder(sess_datas, args)
    np.save(os.path.join(output_dir, f"{file_name}_test_accs.npy"), test_accs)
    if args.shuffle_idx is None: 
        np.save(os.path.join(output_dir, f"{file_name}_models.npy"), models)

        # save pseudo unit IDs
        unit_ids = pd.DataFrame({"PseudoUnitIDs": np.concatenate(sess_datas.apply(lambda x: x.get_pseudo_unit_ids()).values)})
        # unit_ids.to_pickle(os.path.join(output_dir, f"{file_name}_unit_ids.pickle"))
        unit_ids.to_csv(os.path.join(output_dir, f"{file_name}_unit_ids.csv"))

        all_splits = pd.concat(sess_datas.apply(lambda x: x.get_splits_df()).values)
        all_splits.to_pickle(os.path.join(output_dir, f"{file_name}_splits.pickle"))



def process_args(args):
    """
    Determines features, sessions, trial intervals, to use for decoding,
    Adds them to args
    """
    args.feat = FEATURES[args.feat_idx]
    args.trial_interval = get_trial_interval(args.trial_event)
    print(args.beh_filters)
    print(f"Decoding partitions for feat {args.feat} sessions, mode {args.mode}", flush=True)
    print(f"Using {args.fr_type} as inputs", flush=True)
    print(f"With filters {args.beh_filters}", flush=True)
    if args.sig_unit_level:
        print(f"Using only units that are selective with signifance level {args.sig_unit_level}")
    return args


def main(args):
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    args = process_args(args)
    decode(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(BeliefPartitionConfigs(), parser)
    args = parser.parse_args()
    main(args)