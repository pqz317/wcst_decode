
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
from anova_configs import add_defaults_to_parser, AnovaConfigs
import utils.io_utils as io_utils
import utils.anova_utils as anova_utils
import json
from tqdm import tqdm

FEATS_PATH = "/data/patrick_res/sessions/{sub}/feats_at_least_3blocks.pickle"
SESSIONS_PATH = "/data/patrick_res/sessions/{sub}/valid_sessions.pickle"

"""
Script that runs factorial anova per unit on a set of conditions. 
TimeBins is always a default condition
Do analysis a feature at a time, looking for sessions where that feature appears in enough blocks
"""
def load_data(session, args):
    beh = behavioral_utils.load_behavior_from_args(session, args)
    beh = behavioral_utils.get_chosen_single(args.feat, beh)
    beh = behavioral_utils.filter_behavior(beh, args.beh_filters)

    frs = io_utils.get_frs_from_args(args, session)

    if args.time_range is not None: 
        if len(args.time_range) !=2: 
            raise ValueError("must have two ranges")
        start, end = [x / args.trial_interval.interval_size for x in args.time_range]
        frs = frs[(frs.TimeBins >= start) & (frs.TimeBins < end)]
    df = pd.merge(frs, beh, on="TrialNumber")
    return df


def process_session(row, args):
    data = load_data(row.session_name, args)
    all_conds = ["TimeBins"] + args.conditions
    df = anova_utils.anova_factors(data, all_conds)
    unit_vars = df.groupby("PseudoUnitID").apply(lambda x: anova_utils.cal_unit_var(x, all_conds)).reset_index()
    unit_vars["feat"] = args.feat
    return unit_vars

def anova(args):
    file_name = io_utils.get_anova_file_name(args)
    output_dir = io_utils.get_anova_output_dir(args)

    res = pd.concat(args.sessions.progress_apply(lambda x: process_session(x, args), axis=1).values)

    res.to_pickle(os.path.join(output_dir, f"{file_name}.pickle"))


def main():
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(AnovaConfigs(), parser)
    args = parser.parse_args()

    feat_sessions = pd.read_pickle(FEATS_PATH.format(sub=args.subject))
    valid_sess = pd.read_pickle(SESSIONS_PATH.format(sub=args.subject))

    args.feat = FEATURES[args.feat_idx]
    row = feat_sessions[feat_sessions.feat == args.feat].iloc[0]
    args.sessions = valid_sess[valid_sess.session_name.isin(row.sessions)]
    args.trial_interval = get_trial_interval(args.trial_event)

    print(f"Anova for {args.feat} using {len(args.sessions)} sessions, conditions {args.conditions}", flush=True)
    print(f"Using {args.fr_type} as inputs", flush=True)
    print(f"Time ranage: {args.time_range}")
    print(f"With filters {args.beh_filters}", flush=True)
    print(f"Shuffle {args.shuffle_idx} with method {args.shuffle_method}")

    tqdm.pandas()
    anova(args)


if __name__ == "__main__":
    main()