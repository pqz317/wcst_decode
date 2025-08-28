
import os
import numpy as np
import pandas as pd
import utils.behavioral_utils as behavioral_utils

from constants.behavioral_constants import *
from constants.decoding_constants import *

import argparse
from scripts.anova_analysis.anova_configs import add_defaults_to_parser, AnovaConfigs
import utils.io_utils as io_utils
import utils.anova_utils as anova_utils
from tqdm import tqdm

FEATS_PATH = "/data/patrick_res/sessions/{sub}/feats_at_least_3blocks.pickle"
SESSIONS_PATH = "/data/patrick_res/sessions/{sub}/valid_sessions.pickle"

"""
Script that runs factorial anova per unit on a set of conditions. 
TimeBins is always a default condition
Do analysis a feature at a time, looking for sessions where that feature appears in enough blocks
"""
def load_data(session, args, return_merged=True, use_x=False):
    feat = args.feat
    beh = behavioral_utils.load_behavior_from_args(session, args)
    if "feat_pair" in args:
        beh = behavioral_utils.get_belief_partitions_of_pair(beh, args.feat_pair)
    else: 
        feat = args.feat
        beh = behavioral_utils.get_belief_partitions(beh, feat, use_x=use_x)
    beh["Choice"] = beh.apply(lambda x: "Chose" if x[FEATURE_TO_DIM[feat]] == feat else "Not Chose", axis=1)
    beh["FeatPreferred"] = beh["PreferredBelief"].apply(lambda x: "Preferred" if x == feat else "Not Preferred")
    beh = behavioral_utils.filter_behavior(beh, args.beh_filters)
    frs = io_utils.get_frs_from_args(args, session)

    if args.split_idx is not None: 
        print("loading splits, choosing subset of trials")
        splits = pd.read_pickle(io_utils.get_anova_split_path(args))
        row = splits[(splits.sessions == session) & (splits.feat == feat)].iloc[0]
        trials = row[f"split_{args.split_idx}"]
        beh = beh[beh.TrialNumber.isin(trials)]
        frs = frs[frs.TrialNumber.isin(trials)]

    if len(beh) == 0 or len(frs) == 0:
        raise ValueError("no data loaded")
    if return_merged:
        return pd.merge(frs, beh, on="TrialNumber")
    else: 
        return (beh, frs)
    
def run_anova(args, data, all_conds):
    df = anova_utils.anova_factors(data, all_conds)
    unit_vars = df.groupby("PseudoUnitID").apply(lambda x: anova_utils.calc_unit_var(x, all_conds)).reset_index()
    unit_vars = anova_utils.combine_time_fracvar(unit_vars, args.conditions)
    # HACK: don't have a nice way way to compute preference frac var, just adding it here. 
    if "BeliefPartition" in args.conditions: 
        unit_vars[f"x_BeliefPref_comb_time_fracvar"] = unit_vars[f"x_BeliefPartition_comb_time_fracvar"] - unit_vars[f"x_BeliefConf_comb_time_fracvar"]
    if "NextBeliefPartition" in args.conditions: 
        unit_vars[f"x_NextBeliefPref_comb_time_fracvar"] = unit_vars[f"x_NextBeliefPartition_comb_time_fracvar"] - unit_vars[f"x_NextBeliefConf_comb_time_fracvar"]
    unit_vars["feat"] = args.feat    
    return unit_vars


def process_session(row, args):
    data = load_data(row.session_name, args)
    all_conds = ["TimeBins"] + args.conditions
    if args.window_size is None:
        unit_vars = run_anova(args, data, all_conds)
    else:
        unit_vars = []
        data["TimeMilli"] = (data.Time * 1000).round().astype(int)
        pre = -1 * args.trial_interval.pre_interval
        post = args.trial_interval.post_interval
        for window_start in np.arange(pre, post - args.window_size + 1, args.trial_interval.interval_size):
            window_end = window_start + args.window_size
            print(f"Windows: {window_start}, {window_end}")
            window_data = data[(data.TimeMilli >= window_start) & (data.TimeMilli < window_end)]
            print(f"{window_data.Time.nunique()} time points: {window_data.Time.unique()}")
            print("-------")
            window_unit_vars = run_anova(args, window_data, all_conds)
            window_unit_vars["WindowStartMilli"] = window_start
            window_unit_vars["WindowEndMilli"] = window_end
            unit_vars.append(window_unit_vars)
        unit_vars = pd.concat(unit_vars)
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