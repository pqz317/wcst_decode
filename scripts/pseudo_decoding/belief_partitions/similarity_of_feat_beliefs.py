"""
Compute representational similarity between pairs of features, 
"""
import os
import numpy as np
import pandas as pd
import utils.pseudo_classifier_utils as pseudo_classifier_utils
import utils.behavioral_utils as behavioral_utils
import utils.spike_utils as spike_utils
import utils.classifier_utils as classifier_utils

from constants.behavioral_constants import *
from constants.decoding_constants import *
import utils.session_data as session_data

from models.trainer import Trainer
from models.model_wrapper import ModelWrapper
from models.multinomial_logistic_regressor import NormedDropoutMultinomialLogisticRegressor, NormedDropoutNonlinear

import argparse
from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import BeliefPartitionConfigs, add_defaults_to_parser
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
import copy
import json

BOTH_PAIRS_PATH = "/data/patrick_res/sessions/both/pairs_at_least_3blocks_10sess.pickle"

def get_pseudo_frs_for_session(session, args, num_pseudo=100):
    # for grabbing behavior and firing rates, use subject-specific arguments
    # for grabbing decoder weights, use general
    sub_args = copy.deepcopy(args)
    sub_args.subject = behavioral_utils.get_sub_for_session(session)
    print(sub_args.feat)
    beh = behavioral_utils.load_behavior_from_args(session, sub_args)
    beh = behavioral_utils.get_feat_choice_label(beh, sub_args.feat)
    beh = behavioral_utils.get_belief_partitions(beh, sub_args.feat, use_x=True)

    sub_beh = beh[beh.BeliefPartition == "High X"]

    frs = spike_utils.get_frs_from_args(sub_args, session)
    frs["TimeIdx"] = (frs["Time"] * 10).round().astype(int)

    sub_frs = frs[frs.TrialNumber.isin(sub_beh.TrialNumber)]

    rng = np.random.default_rng()
    trial_nums = rng.choice(sub_frs.TrialNumber.unique(), num_pseudo)
    pseudo_trials = pd.DataFrame({"TrialNumber": trial_nums, "PseudoTrialNumber": list(range(num_pseudo))})
    pseudo_frs = pd.merge(sub_frs, pseudo_trials, on="TrialNumber")
    pseudo_frs["session"] = session
    return pseudo_frs

def get_sims(pair, args):
    (feat1, feat2) = pair.pair
    args.feat = feat1
    feat1_res = pd.concat(pd.Series(pair.sessions).apply(lambda x: get_pseudo_frs_for_session(x, args)).values)

    args.feat = feat2
    feat2_res = pd.concat(pd.Series(pair.sessions).apply(lambda x: get_pseudo_frs_for_session(x, args)).values)

    merged = pd.merge(feat1_res, feat2_res, on=["PseudoUnitID", "PseudoTrialNumber", "TimeIdx"], suffixes=["_feat1", "_feat2"], how="outer").fillna(0)

    sims = merged.groupby(["PseudoTrialNumber", "TimeIdx"]).apply(lambda x: classifier_utils.cosine_sim(x.FiringRate_feat1, x.FiringRate_feat2)).reset_index(name="cosine_sim")
    return sims

def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(BeliefPartitionConfigs(), parser)
    args = parser.parse_args()

    pairs = pd.read_pickle(BOTH_PAIRS_PATH).reset_index(drop=True)
    pair_row = pairs.iloc[args.pair_idx]
    args.all_sessions = pd.DataFrame({"session_name": pair_row.sessions})
    args.trial_interval = get_trial_interval(args.trial_event)
    sims = get_sims(pair_row, args)

    args.feat_pair = pair_row.pair
    args.base_output_path = "/data/patrick_res/belief_similarities"
    file_name = belief_partitions_io.get_ccgp_file_name(args)
    output_dir = belief_partitions_io.get_dir_name(args)
    sims.to_pickle(os.path.join(output_dir, f"{file_name}_sims.pickle"))
    
if __name__ == "__main__":
    main()