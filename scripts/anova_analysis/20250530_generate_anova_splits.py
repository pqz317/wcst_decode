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
from distutils.util import strtobool

from tqdm import tqdm

FEATS_PATH = "/data/patrick_res/sessions/{sub}/feats_at_least_3blocks.pickle"

def split_by_condition(group):
    rng = np.random.default_rng()
    trials = group.TrialNumber.unique()
    rng.shuffle(trials)
    split_point = len(trials) // 2
    return pd.Series({"split_0": trials[:split_point], "split_1": trials[split_point:]})

def find_trial_splits(args, row):
    args.feat = row.feat
    feat = args.feat
    beh = behavioral_utils.load_behavior_from_args(row.sessions, args)
    beh = behavioral_utils.get_belief_partitions(beh, feat)
    beh["Choice"] = beh.apply(lambda x: "Chose" if x[FEATURE_TO_DIM[feat]] == feat else "Not Chose", axis=1)
    beh["FeatPreferred"] = beh["PreferredBelief"].apply(lambda x: "Preferred" if x == feat else "Not Preferred")
    beh = behavioral_utils.filter_behavior(beh, args.beh_filters)
    cond_splits = beh.groupby("BeliefPartition").apply(split_by_condition).reset_index()
    return pd.Series({
        "split_0": np.concatenate(cond_splits.split_0.values), 
        "split_1": np.concatenate(cond_splits.split_1.values)
    })

def main():
    tqdm.pandas()
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(AnovaConfigs(), parser)
    parser.add_argument('--dry_run', default=True, type=lambda x: bool(strtobool(x)))

    args = parser.parse_args()

    output_path = io_utils.get_anova_split_path(args)
    print(f"Will save results to {output_path}")


    feat_sessions = pd.read_pickle(FEATS_PATH.format(sub=args.subject))
    feat_sess_pairs = feat_sessions.explode("sessions")
    feat_sess_pairs[["split_0", "split_1"]] = feat_sess_pairs.progress_apply(lambda x: find_trial_splits(args, x), axis=1)

    if not args.dry_run:
        print("Dry run is false, saving splits")
        feat_sess_pairs.to_pickle(output_path)



if __name__ == "__main__":
    main()