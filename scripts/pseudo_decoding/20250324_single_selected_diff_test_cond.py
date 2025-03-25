"""
Want to see if splitting trials during pref/notpref during test yields different test accuracies
Avoids (I think) issue with decoder picking up on time confounds. 
Steps: 
- Load session data for chosen condition. 
- Train chosen decoders all the same
- Grab session datas, grab models, create new session datas that filter on pref/notpref
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
from decode_single_selected_features import FEATS_PATH, SESSIONS_PATH, UNITS_PATH
from itertools import cycle


def generate_test_sess_data_for_cond(sess_name, feat, beh, frs, model_splits, cond):
    cond_splits = []
    for split in model_splits:
        # find positive condition trials that are preferred
        test_trials = np.concatenate(split.TestTrials.values)
        if cond == "pref":
            cond_beh = behavioral_utils.get_chosen_preferred_single(feat, beh)
        elif cond == "not_pref":
            cond_beh = behavioral_utils.get_chosen_not_preferred_single(feat, beh)
        else: 
            raise ValueError()
        cond_beh = cond_beh[cond_beh.TrialNumber.isin(test_trials)]
        cond_beh = behavioral_utils.balance_trials_by_condition(cond_beh, ["Choice"])
        cond_df = cond_beh.groupby("Choice").apply(lambda g: g.TrialNumber.unique()).reset_index(name="TestTrials")
        cond_df = cond_df.rename(columns={"Choice": "Condition"})
        # add an empty train trials list per cond
        cond_df["TrainTrials"] = [[] for _ in range(len(cond_df))]
        cond_splits.append(cond_df)
    splitter = cycle(cond_splits)
    sess_data = SessionData(sess_name, beh, frs, splitter)
    sess_data.pre_generate_splits(len(model_splits))
    return sess_data

def load_session_data(row, region_units, args):
    """
    Loads session datas from behavior, but returns a tuple of sess datas, (chosen, pref, not_pref)
    - chosen: data used to train the model
    - pref/not_pref data that only includes test data in the chosen splits, split by pref/not_pref
    """
    sess_name = row.session_name
    feat = args.feat

    behavior_path = SESS_BEHAVIOR_PATH.format(sub=args.subject, sess_name=sess_name)
    beh = pd.read_csv(behavior_path)
    beh = behavioral_utils.get_valid_trials(beh)
    feature_selections = behavioral_utils.get_selection_features(beh)
    beh = pd.merge(beh, feature_selections, on="TrialNumber", how="inner")
    beh = behavioral_utils.get_beliefs_per_session(beh, sess_name)
    beh = behavioral_utils.get_belief_value_labels(beh)
    beh = behavioral_utils.filter_behavior(beh, {"PreferredChosen": True})

    # shift TrialNumbers by some random amount
    if args.shuffle_idx is not None: 
        beh = behavioral_utils.shuffle_beh_by_shift(beh, buffer=50, seed=args.shuffle_idx)

    chosen_beh = behavioral_utils.get_chosen_single(feat, beh)
    chosen_beh = behavioral_utils.balance_trials_by_condition(chosen_beh, ["Choice"])

    frs = io_utils.get_frs_from_args(args, sess_name)
    frs = frs.rename(columns={"FiringRate": "Value"})
    frs["PseudoUnitID"] = int(sess_name) * 100 + frs.UnitID.astype(int)
    if region_units is not None: 
        frs = frs[frs.PseudoUnitID.isin(region_units)]
    if len(frs) == 0 or len(chosen_beh) == 0:
        return None
    splitter = ConditionTrialSplitter(chosen_beh, "Choice", args.test_ratio)
    chosen_sess_data = SessionData(sess_name, chosen_beh, frs, splitter)
    splits = chosen_sess_data.pre_generate_splits(args.num_splits)

    pref_sess_data = generate_test_sess_data_for_cond(sess_name, feat, beh, frs, splits, "pref")
    not_pref_sess_data = generate_test_sess_data_for_cond(sess_name, feat, beh, frs, splits, "not_pref")
    return chosen_sess_data, pref_sess_data, not_pref_sess_data



def decode_chosen(args, sess_datas, time_bins):

    classes = [args.feat, "other"]
    num_neurons = sess_datas.apply(lambda x: x.get_num_neurons()).sum()
    init_params = {"n_inputs": num_neurons, "p_dropout": 0.5, "n_classes": len(classes)}
    # create a trainer object
    trainer = Trainer(learning_rate=0.05, max_iter=500, batch_size=1000)
    # create a wrapper for the decoder
    model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, classes)

    # calculate time bins (in seconds)
    _, test_accs, _, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(
        model, sess_datas, time_bins, NUM_SPLITS, NUM_TRAIN_PER_COND, NUM_TEST_PER_COND, use_v2=args.use_v2_pseudo
    )
    return test_accs, models


def decode(args):
    region_units = spike_utils.get_region_units(args.region_level, args.regions, UNITS_PATH.format(sub=args.subject))
    sessions = args.sessions


    sess_datas = sessions.apply(lambda row: load_session_data(
        row, region_units, args
    ), axis=1)
    sess_datas = sess_datas.dropna()

    chosen_sess_datas = sess_datas.apply(lambda x: x[0])
    trial_interval = args.trial_interval
    time_bins = np.arange(0, (trial_interval.post_interval + trial_interval.pre_interval) / 1000, trial_interval.interval_size / 1000)

    chosen_test_accs, models = decode_chosen(args, chosen_sess_datas, time_bins)

    pref_sess_datas = sess_datas.apply(lambda x: x[1])
    pref_test_accs = pseudo_classifier_utils.evaluate_model_with_data(models, pref_sess_datas, time_bins)

    not_pref_sess_datas = sess_datas.apply(lambda x: x[2])
    not_pref_test_accs = pseudo_classifier_utils.evaluate_model_with_data(models, not_pref_sess_datas, time_bins)  

    # naming for files, directory
    file_name = io_utils.get_selected_features_file_name(args)
    output_dir = io_utils.get_selected_features_output_dir(args)

        # np.save(os.path.join(output_dir, f"{file_name}_train_accs.npy"), train_accs)
    np.save(os.path.join(output_dir, f"{file_name}_chosen_test_accs.npy"), chosen_test_accs)
    np.save(os.path.join(output_dir, f"{file_name}_pref_test_accs.npy"), pref_test_accs)
    np.save(os.path.join(output_dir, f"{file_name}_not_pref_test_accs.npy"), not_pref_test_accs)

    # np.save(os.path.join(output_dir, f"{file_name}_shuffled_accs.npy"), shuffled_accs)
    if args.shuffle_idx is None: 
        np.save(os.path.join(output_dir, f"{file_name}_chosen_models.npy"), models)


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
    args.base_output_path = "/data/patrick_res/single_selected_diff_test_cond"

    print(f"Decoding choosing {args.feat} vs not using {len(args.sessions)} sessions, condition {args.condition}", flush=True)
    print(f"Using {args.fr_type} as inputs", flush=True)
    print(f"With filters {args.beh_filters}", flush=True)
    if args.use_v2_pseudo: 
        print(f"Using new pseudo data generation")
    decode(args)


if __name__ == "__main__":
    main()