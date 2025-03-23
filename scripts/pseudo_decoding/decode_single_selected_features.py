"""
Script to decode selected features, but in a binary fashion
chosen vs. not chosen, under different conditions: 
- chosen preferred vs. not chosen, not preferred
- chosen not preferred vs. not chosen not preferred
- chosen vs. not chosen, including preferred not preferred
Additionally, include filters for response (correct/inc), values (low/high). 
When a filter is included, balance across other conditions in the filter as well. 
Going to see if this analysis can replace 20241113_decode_preferred_beliefs.py
Motivations here: 
https://www.notion.so/walkerlab/Potential-Decoder-weights-analyses-1b22dc9f99928041aad6d0e0e79be875
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
from single_selected_feature_configs import add_defaults_to_parser, SingleSelectedFeatureConfigs
import utils.io_utils as io_utils
import json

FEATS_PATH = "/data/patrick_res/sessions/SA/feats_at_least_3blocks.pickle"
SESSIONS_PATH = "/data/patrick_res/sessions/SA/valid_sessions.pickle"
UNITS_PATH = "/data/patrick_res/firing_rates/{sub}/all_units.pickle"
SESS_BEHAVIOR_PATH = "/data/patrick_res/behavior/SA/{sess_name}_object_features.csv"

DATA_MODE = "FiringRate"

def load_session_data(row, region_units, args):
    sess_name = row.session_name
    feat = args.feat

    behavior_path = SESS_BEHAVIOR_PATH.format(sess_name=sess_name)
    beh = pd.read_csv(behavior_path)
    beh = behavioral_utils.get_valid_trials(beh)
    feature_selections = behavioral_utils.get_selection_features(beh)
    beh = pd.merge(beh, feature_selections, on="TrialNumber", how="inner")
    beh = behavioral_utils.get_beliefs_per_session(beh, sess_name)
    beh = behavioral_utils.get_belief_value_labels(beh)
    beh = behavioral_utils.get_prev_choice_fbs(beh)
    if args.balance_by_filters: 
        beh = behavioral_utils.balance_trials_by_condition(beh, list(args.beh_filters.keys()))
    beh = behavioral_utils.filter_behavior(beh, args.beh_filters)

    # shift TrialNumbers by some random amount
    if args.shuffle_idx is not None: 
        beh = behavioral_utils.shuffle_beh_by_shift(beh, buffer=50, seed=args.shuffle_idx)

    choice = behavioral_utils.get_chosen_single(feat, beh)
    pref = behavioral_utils.get_chosen_preferred_single(feat, beh)
    not_pref = behavioral_utils.get_chosen_not_preferred_single(feat, beh)

    # balance the conditions out:
    # use minimum number of trials between the chosen preferred, chosen not preferred conditions
    # HACK: don't use balance across conditions if already balancing by filters
    if not args.balance_by_filters:
        min_trials = np.min((
            np.min(choice.groupby("Choice").count().TrialNumber),
            np.min(pref.groupby("Choice").count().TrialNumber),
            np.min(not_pref.groupby("Choice").count().TrialNumber)
        ))
    else: 
        min_trials = None

    if args.condition == "chosen": 
        sub_beh = choice
    elif args.condition == "pref": 
        sub_beh = pref
    elif args.condition == "not_pref":
        sub_beh = not_pref
    elif args.condition == "pref_vs_not_pref":
        # NOTE: kinda hacky but this allows the script to be run for pref vs. not pref in the same way
        sub_beh = pd.concat([
            pref[pref.Choice == feat],
            not_pref[not_pref.Choice == feat]
        ])
        sub_beh["Choice"] = sub_beh.PreferredBelief.apply(lambda x: "pref" if x == feat else "not_pref")
    else: 
        raise ValueError(f"invalid condition flag {args.condition}")
    
    sub_beh = behavioral_utils.balance_trials_by_condition(sub_beh, ["Choice"], min=min_trials)
    # print(f"number of green trials left after balancing everything: {len(sub_beh[sub_beh[FEATURE_TO_DIM[feat]] == feat])}")


    frs = io_utils.get_frs_from_args(args, sess_name)
    frs = frs.rename(columns={DATA_MODE: "Value"})
    frs["PseudoUnitID"] = int(sess_name) * 100 + frs.UnitID.astype(int)
    if region_units is not None: 
        frs = frs[frs.PseudoUnitID.isin(region_units)]
    if len(frs) == 0 or len(sub_beh) == 0:
        return None
    splitter = ConditionTrialSplitter(sub_beh, "Choice", args.test_ratio)
    session_data = SessionData(sess_name, sub_beh, frs, splitter)
    session_data.pre_generate_splits(args.num_splits)
    return session_data



def decode(args):
    region_units = spike_utils.get_region_units(args.region_level, args.regions, UNITS_PATH.format(sub=args.subject))
    trial_interval = args.trial_interval
    sessions = args.sessions

    # naming for files, directory
    file_name = io_utils.get_selected_features_file_name(args)
    output_dir = io_utils.get_selected_features_output_dir(args)

    # load up session data to train network
    sess_datas = sessions.apply(lambda row: load_session_data(
        row, region_units, args
    ), axis=1)
    sess_datas = sess_datas.dropna()

    # save pseudo unit IDs
    unit_ids = pd.DataFrame({"PseudoUnitIDs": np.concatenate(sess_datas.apply(lambda x: x.get_pseudo_unit_ids()).values)})
    # unit_ids.to_pickle(os.path.join(output_dir, f"{file_name}_unit_ids.pickle"))
    unit_ids.to_csv(os.path.join(output_dir, f"{file_name}_unit_ids.csv"))


    # train the network
    # setup decoder, specify all possible label classes, number of neurons, parameters
    # NOTE: hacky as well, making pref vs not pref work here
    classes = ["pref", "not_pref"] if args.condition == "pref_vs_not_pref" else [args.feat, "other"]
    num_neurons = sess_datas.apply(lambda x: x.get_num_neurons()).sum()
    init_params = {"n_inputs": num_neurons, "p_dropout": 0.5, "n_classes": len(classes)}
    # create a trainer object
    trainer = Trainer(learning_rate=0.05, max_iter=500, batch_size=1000)
    # create a wrapper for the decoder
    model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, classes)

    # calculate time bins (in seconds)
    time_bins = np.arange(0, (trial_interval.post_interval + trial_interval.pre_interval) / 1000, trial_interval.interval_size / 1000)
    _, test_accs, _, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(
        model, sess_datas, time_bins, NUM_SPLITS, NUM_TRAIN_PER_COND, NUM_TEST_PER_COND, use_v2=args.use_v2_pseudo
    )

    # np.save(os.path.join(output_dir, f"{file_name}_train_accs.npy"), train_accs)
    np.save(os.path.join(output_dir, f"{file_name}_test_accs.npy"), test_accs)
    # np.save(os.path.join(output_dir, f"{file_name}_shuffled_accs.npy"), shuffled_accs)
    if args.shuffle_idx is None: 
        np.save(os.path.join(output_dir, f"{file_name}_models.npy"), models)


def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(SingleSelectedFeatureConfigs(), parser)
    args = parser.parse_args()

    if args.subject == "SA": 
        feat_sessions = pd.read_pickle(FEATS_PATH)
        valid_sess = pd.read_pickle(SESSIONS_PATH)

    else: 
        raise ValueError("unsupported subject")
    args.feat = FEATURES[args.feat_idx]
    row = feat_sessions[feat_sessions.feat == args.feat].iloc[0]
    args.sessions = valid_sess[valid_sess.session_name.isin(row.sessions)]
    args.trial_interval = get_trial_interval(args.trial_event)

    print(args.beh_filters)
    print(f"Decoding choosing {args.feat} vs not using {len(args.sessions)} sessions, condition {args.condition}", flush=True)
    print(f"Using {args.fr_type} as inputs", flush=True)
    print(f"With filters {args.beh_filters}", flush=True)
    if args.use_v2_pseudo: 
        print(f"Using new pseudo data generation")
    decode(args)


if __name__ == "__main__":
    main()