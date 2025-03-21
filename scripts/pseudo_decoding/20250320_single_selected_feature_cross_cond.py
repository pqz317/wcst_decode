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
from single_selected_feature_configs import add_defaults_to_parser, SingleSelectedFeatureCrossCondConfigs
import utils.io_utils as io_utils
import json
from decode_single_selected_features import load_session_data, FEATS_PATH, SESSIONS_PATH, UNITS_PATH

def cross_cond_decode(args):
    region_units = spike_utils.get_region_units(args.region_level, args.regions, UNITS_PATH.format(sub=args.subject))
    trial_interval = args.trial_interval
    sessions = args.sessions

    # naming for files, directory
    args.condition = args.model_cond
    model_file_name = io_utils.get_selected_features_file_name(args)
    output_dir = io_utils.get_selected_features_output_dir(args)
    model = np.load(os.path.join(output_dir, f"{model_file_name}_models.npy"), allow_pickle=True)

    args.condition = args.data_cond
    # load up session data to train network
    sess_datas = sessions.apply(lambda row: load_session_data(
        row, region_units, args
    ), axis=1)
    sess_datas = sess_datas.dropna()

    # calculate time bins (in seconds)
    time_bins = np.arange(0, (trial_interval.post_interval + trial_interval.pre_interval) / 1000, trial_interval.interval_size / 1000)
    accs = pseudo_classifier_utils.evaluate_model_with_data(model, sess_datas, time_bins)

    save_file_name = io_utils.get_selected_features_cross_cond_file_name(args)
    np.save(os.path.join(output_dir, f"{save_file_name}_accs.npy"), accs)


def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(SingleSelectedFeatureCrossCondConfigs(), parser)
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

    print(f"Decoding choosing {args.feat} vs not using {len(args.sessions)} sessions, using {args.model_cond} models to decode {args.data_cond} data", flush=True)
    print(f"Using {args.fr_type} as inputs", flush=True)
    print(f"With filters {args.beh_filters}", flush=True)
    if args.use_v2_pseudo: 
        print(f"Using new pseudo data generation")
    cross_cond_decode(args)


if __name__ == "__main__":
    main()