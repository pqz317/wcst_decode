"""
Mainly a copy of 20240725_high_conf_max_feat_by_pairs.py and 20240725_high_conf_max_feat_by_pairs.py
with some changes to make it beliefs. 
Also add a flag to combine the two scripts
"""

import os
import numpy as np
import pandas as pd
import utils.pseudo_utils as pseudo_utils
import utils.pseudo_classifier_utils as pseudo_classifier_utils
import utils.behavioral_utils as behavioral_utils
import utils.spike_utils as spike_utils

from constants.behavioral_constants import *
from constants.decoding_constants import *
from utils.session_data import SessionData

from models.trainer import Trainer
from models.model_wrapper import ModelWrapper, ModelWrapperLinearRegression
from models.multinomial_logistic_regressor import NormedDropoutMultinomialLogisticRegressor
from trial_splitters.condition_trial_splitter import ConditionTrialSplitter 

import argparse
from scripts.pseudo_decoding.preferred_beliefs_by_pairs_configs import add_defaults_to_parser
import utils.io_utils as io_utils
import json

# the output directory to store the data
OUTPUT_DIR = "/data/res/pseudo"
# path to a dataframe of sessions to analyze
SESSIONS_PATH = "/data/patrick_res/sessions/SA/valid_sessions.pickle"
# PAIRS_PATH = "/data/patrick_res/sessions/SA/pairs_at_least_3blocks_7sess.pickle"
# MIN_TRIALS_FOR_PAIRS_PATH = "/data/patrick_res/sessions/SA/pairs_at_least_3blocks_7sess_min_trials.pickle"

PAIRS_PATH = "/data/patrick_res/sessions/SA/pairs_at_least_3blocks_10sess_more_sess.pickle"

# SESSIONS_PATH = "/data/patrick_res/sessions/valid_sessions_rpe.pickle"
# PAIRS_PATH = "/data/patrick_res/sessions/pairs_at_least_3blocks_10sess.pickle"

# path for each session, specifying behavior
SESS_BEHAVIOR_PATH = "/data/patrick_res/behavior/SA/{sess_name}_object_features.csv"
# path for each session, for spikes that have been pre-aligned to event time and binned. 

UNITS_PATH = "/data/patrick_res/firing_rates/{sub}/all_units.pickle"

DATA_MODE = "FiringRate"


def load_session_data(row, sub_units, args):
    sess_name = row.session_name

    beh = behavioral_utils.load_behavior_from_args(sess_name, args)

    pair = args.feat_pair

    if args.balance_by_filters: 
        beh = behavioral_utils.balance_trials_by_condition(beh, list(args.beh_filters.keys()))
    beh = behavioral_utils.filter_behavior(beh, args.beh_filters)

    not_pref = behavioral_utils.get_chosen_not_preferred_trials(pair, beh, args.high_val_only)
    pref = behavioral_utils.get_chosen_preferred_trials(pair, beh, args.high_val_only)

    # balance the conditions out:
    # use minimum number of trials between the chosen preferred, chosen not preferred conditions
    min_trials = np.min((
        np.min(pref.groupby("Choice").count().TrialNumber),
        np.min(not_pref.groupby("Choice").count().TrialNumber)
    ))
    sub_beh = not_pref if args.chosen_not_preferred else pref
    sub_beh = behavioral_utils.balance_trials_by_condition(sub_beh, ["Choice"], min=min_trials)

    frs = io_utils.get_frs_from_args(args, sess_name)
    frs = frs.rename(columns={DATA_MODE: "Value"})
    if sub_units is not None: 
        frs = frs[frs.PseudoUnitID.isin(sub_units)]
    if len(frs) == 0 or len(sub_beh) == 0:
        return None
    splitter = ConditionTrialSplitter(sub_beh, "Choice", args.test_ratio)
    session_data = SessionData(sess_name, sub_beh, frs, splitter)
    session_data.pre_generate_splits(args.num_splits)
    return session_data

def decode(args):
    sub_units = spike_utils.get_region_units(args.region_level, args.regions, UNITS_PATH.format(sub=args.subject))
    sub_units = spike_utils.get_sig_units(args, sub_units)
    trial_interval = args.trial_interval
    sessions = args.sessions

    # naming for files, directory
    file_name = io_utils.get_preferred_beliefs_file_name(args)
    output_dir = io_utils.get_preferred_beliefs_output_dir(args)

    # load up session data to train network
    pair = args.feat_pair
    sess_datas = sessions.apply(lambda row: load_session_data(
        row, sub_units, args
    ), axis=1)
    sess_datas = sess_datas.dropna()

    # save pseudo unit IDs
    unit_ids = pd.DataFrame({"PseudoUnitIDs": np.concatenate(sess_datas.apply(lambda x: x.get_pseudo_unit_ids()).values)})
    # unit_ids.to_pickle(os.path.join(output_dir, f"{file_name}_unit_ids.pickle"))
    unit_ids.to_csv(os.path.join(output_dir, f"{file_name}_unit_ids.csv"))


    # train the network
    # setup decoder, specify all possible label classes, number of neurons, parameters
    classes = [pair[0], pair[1]]
    num_neurons = sess_datas.apply(lambda x: x.get_num_neurons()).sum()
    init_params = {"n_inputs": num_neurons, "p_dropout": 0.5, "n_classes": len(classes)}
    # create a trainer object
    trainer = Trainer(learning_rate=0.05, max_iter=500, batch_size=1000)
    # create a wrapper for the decoder
    model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, classes)

    # calculate time bins (in seconds)
    time_bins = np.arange(0, (trial_interval.post_interval + trial_interval.pre_interval) / 1000, trial_interval.interval_size / 1000)
    train_accs, test_accs, shuffled_accs, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(
        model, sess_datas, time_bins, NUM_SPLITS, NUM_TRAIN_PER_COND, NUM_TEST_PER_COND
    )

    np.save(os.path.join(output_dir, f"{file_name}_train_accs.npy"), train_accs)
    np.save(os.path.join(output_dir, f"{file_name}_test_accs.npy"), test_accs)
    np.save(os.path.join(output_dir, f"{file_name}_shuffled_accs.npy"), shuffled_accs)
    np.save(os.path.join(output_dir, f"{file_name}_models.npy"), models)


def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(parser)
    args = parser.parse_args()

    if args.subject == "SA": 
        pairs = pd.read_pickle(PAIRS_PATH)
        valid_sess = pd.read_pickle(SESSIONS_PATH)
    else: 
        raise ValueError("unsupported subject")
    pair_row = pairs.iloc[args.pair_idx]
    args.feat_pair = pair_row.pair
    args.sessions = valid_sess[valid_sess.session_name.isin(pair_row.sessions)]
    args.trial_interval = get_trial_interval(args.trial_event)

    print(f"Decoding between {args.feat_pair} using between {len(args.sessions)} sessions, chosen not preferred {args.chosen_not_preferred}", flush=True)
    print(f"Using {args.fr_type} as inputs", flush=True)
    decode(args)


if __name__ == "__main__":
    main()