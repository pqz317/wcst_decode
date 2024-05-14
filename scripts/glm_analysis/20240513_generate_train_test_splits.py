# A GLM for looking at the max valued feature
# Only look at trials where: 
# max feature matches current rule, and trial is correct (current rule is chosen)
# only look at blocks where at rule shows up at least 2 times
# look at neural activity relative to mean of block (mean subtracted) 
# Evaluate test score against a shuffled dataset, shuffling rule labels by block

# Simpler model for value representation, just look at total number of spikes in inter-trial
# don't worry about residuals or any other regressors, just value

import argparse
import os
import numpy as np
import pandas as pd
import utils.behavioral_utils as behavioral_utils
import utils.information_utils as information_utils
import utils.spike_utils as spike_utils
import utils.io_utils as io_utils
import utils.glm_utils as glm_utils
from matplotlib import pyplot as plt
import time
from constants.glm_constants import *
from constants.behavioral_constants import *
from trial_splitters.kfold_splitter import KFoldSplitter



# the output directory to store the data
OUTPUT_DIR = "/data/patrick_res"
SESSIONS_PATH = "/data/valid_sessions_rpe.pickle"
SESS_BEHAVIOR_PATH = "/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
SESSIONS_PATH = "/data/patrick_res/sessions/valid_sessions_rpe.pickle"
# path for each session, for spikes that have been pre-aligned to event time and binned. 

# formatting will replace 'feedback_type' first, then replace others in another function
FEATURE_DIMS = ["Color", "Shape", "Pattern"]

MIN_BLOCKS_PER_RULE = 2

SEED = 42
N_SPLITS = 5

def generate_splits(row):
    session = row.session_name
    behavior_path = SESS_BEHAVIOR_PATH.format(sess_name=session)
    beh = pd.read_csv(behavior_path)

    # filter trials 
    beh = behavioral_utils.get_valid_trials(beh)
    feature_selections = behavioral_utils.get_selection_features(beh)
    beh = pd.merge(beh, feature_selections, on="TrialNumber")
    beh = behavioral_utils.get_feature_values_per_session(session, beh)
    beh = behavioral_utils.filter_blocks_by_rule_occurence(beh, MIN_BLOCKS_PER_RULE)
    beh = behavioral_utils.filter_max_feat_correct(beh)

    splitter = KFoldSplitter(beh.TrialNumber.values, N_SPLITS, SEED)
    rows = []
    for i in range(N_SPLITS):
        train, test = next(splitter)
        row = {"session": session, "split_idx": i, "train": train, "test": test}
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    splits = pd.concat(valid_sess.apply(generate_splits, axis=1).values)
    splits.to_pickle(os.path.join(OUTPUT_DIR, "max_feat_kfold_splits.pickle"))

if __name__ == "__main__":
    main()

