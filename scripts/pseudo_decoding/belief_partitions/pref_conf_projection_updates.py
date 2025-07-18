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
from models.multinomial_logistic_regressor import NormedDropoutMultinomialLogisticRegressor, NormedDropoutNonlinear

import argparse
from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import BeliefPartitionConfigs, add_defaults_to_parser
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
import copy
import json

MODE_TO_DIRECTION_LABELS = {
    "pref": {"high": "High X", "low": "High Not X"},
    "conf": {"high": "High", "low": "Low"}
}

def load_pref_vector(args):
    # TODO: need to clean this up
    high_idx = MODE_TO_CLASSES[args.mode].index(MODE_TO_DIRECTION_LABELS[args.mode]["high"])
    low_idx = MODE_TO_CLASSES[args.mode].index(MODE_TO_DIRECTION_LABELS[args.mode]["low"])

    models = belief_partitions_io.read_models(args, [args.feat])
    unit_ids = belief_partitions_io.read_units(args, [args.feat])
    models["weightsdiff"] = models.apply(lambda x: x.models.coef_[high_idx, :] - x.models.coef_[low_idx, :], axis=1)
    models["batch_mean"] = models.apply(lambda x: x.models.model.norm.running_mean.detach().cpu().numpy(), axis=1)
    # 1e-5 from torch batchnorm1d, numerical 
    models["batch_std"] = models.apply(lambda x: np.sqrt(x.models.model.norm.running_var.detach().cpu().numpy() + 1e-5), axis=1)

    def avg_and_label(x):
        weights_diff_means = np.mean(np.vstack(x.weightsdiff.values), axis=0)
        mean_means = np.mean(np.vstack(x.batch_mean.values), axis=0)
        std_means = np.mean(np.vstack(x.batch_std.values), axis=0)
        pos = np.arange(len(weights_diff_means))
        
        return pd.DataFrame({"pos": pos, "weightsdiff": weights_diff_means, "mean": mean_means, "std": std_means})
    weights = models.groupby(["Time", "feat"]).apply(avg_and_label).reset_index()
    weights = pd.merge(weights, unit_ids, on=["feat", "pos"])
    # weights times are right aligned, but firing rates are left-aligned... 
    # at some point need to reconcile the two
    weights["TimeIdx"] = ((weights["Time"] - 0.1) * 10).round().astype(int)
    return weights

def get_proj_pseudo_for_session(session, args, num_pseudo=1000):
    # for grabbing behavior and firing rates, use subject-specific arguments
    # for grabbing decoder weights, use general
    sub_args = copy.deepcopy(args)
    sub_args.subject = behavioral_utils.get_sub_for_session(session)

    beh = behavioral_utils.load_behavior_from_args(session, sub_args)
    beh["NextTrialNumber"] = beh.shift(-1).TrialNumber
    beh = beh[~beh.NextTrialNumber.isna()]
    beh["NextTrialNumber"] = beh.NextTrialNumber.astype(int)
    beh = behavioral_utils.get_feat_choice_label(beh, sub_args.feat)
    beh = behavioral_utils.get_belief_partitions(beh, sub_args.feat, use_x=True)

    beh = behavioral_utils.filter_behavior(beh, sub_args.conditions)

    frs = spike_utils.get_frs_from_args(sub_args, session)
    frs["TimeIdx"] = (frs["Time"] * 10).round().astype(int)
    fr_w_next = pd.merge(frs, beh[["TrialNumber", "NextTrialNumber"]], on="TrialNumber")
    fr_w_next = pd.merge(fr_w_next, frs, left_on=["NextTrialNumber", "PseudoUnitID", "TimeIdx"], right_on=["TrialNumber", "PseudoUnitID", "TimeIdx"], suffixes=[None, "Next"])
    fr_w_next["FiringRateDiff"] = fr_w_next["FiringRateNext"] - fr_w_next["FiringRate"]
    fr_diffs = fr_w_next[["TrialNumber", "PseudoUnitID", "TimeIdx", "FiringRateDiff"]]


    model_args = copy.deepcopy(args)
    # ensure models are not from shuffles
    model_args.shuffle_idx = None
    weights = load_pref_vector(model_args)
    proj = pd.merge(fr_diffs, weights, on=["PseudoUnitID", "TimeIdx"])

    if len(proj) == 0: 
        # no projection to consider
        return None

    def compute_dot(group):
        # currently using variance found from batch norm layer
        return (group.FiringRateDiff / group["std"] * group.weightsdiff).sum()

    proj = proj.groupby(["TimeIdx", "TrialNumber"]).apply(compute_dot).reset_index(name="proj")
    rng = np.random.default_rng()
    trial_nums = rng.choice(proj.TrialNumber.unique(), num_pseudo)
    pseudo_trials = pd.DataFrame({"TrialNumber": trial_nums, "PseudoTrialNumber": list(range(num_pseudo))})
    proj_pseudo = pd.merge(proj, pseudo_trials, on="TrialNumber")
    proj_pseudo["session"] = session

    return proj_pseudo

def proj_all_sessions(args, sessions): 
    res = []
    for session in sessions:
        proj = get_proj_pseudo_for_session(session, args)
        if proj is not None: 
            res.append(proj)
    res = pd.concat(res)
    summed_proj = res.groupby(["TimeIdx", "PseudoTrialNumber"]).proj.sum().reset_index(name="proj")
    return summed_proj


def get_feat_sessions_for_sub(args):
    valid_sess = pd.read_pickle(SESSIONS_PATH.format(sub=args.subject))
    feats = pd.read_pickle(FEATS_PATH.format(sub=args.subject))
    feat_sessions = feats[feats.feat == args.feat].iloc[0].sessions
    return valid_sess[valid_sess.session_name.isin(feat_sessions)]

def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(BeliefPartitionConfigs(), parser)
    parser.add_argument(f'--conditions', default=None, type=lambda x: json.loads(x))

    args = parser.parse_args()
    args.trial_interval = get_trial_interval(args.trial_event)
    args.feat = FEATURES[args.feat_idx]

    if args.subject == "both":
        sa_args = copy.deepcopy(args)
        sa_args.subject = "SA"
        sa_sessions = get_feat_sessions_for_sub(sa_args)
        
        bl_args = copy.deepcopy(args)
        bl_args.subject = "BL"
        bl_sessions = get_feat_sessions_for_sub(bl_args)
        valid_sess = pd.concat((sa_sessions, bl_sessions))
    else: 
        valid_sess = get_feat_sessions_for_sub(args)
    
    args.all_sessions = valid_sess
    summed_proj = proj_all_sessions(args, valid_sess.session_name)
    save_args = copy.deepcopy(args)
    # hack, just leverage beh filters for the conditions here. 
    save_args.beh_filters = save_args.conditions
    save_args.base_output_path = "/data/patrick_res/update_projections"
    out_dir = belief_partitions_io.get_dir_name(save_args)
    file_name = belief_partitions_io.get_file_name(save_args)
    summed_proj.to_pickle(os.path.join(out_dir, f"{file_name}_projections.pickle"))



if __name__ == "__main__":
    main()