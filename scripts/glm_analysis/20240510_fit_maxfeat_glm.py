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


# the output directory to store the data
OUTPUT_DIR = "/data/res"
SESSIONS_PATH = "/data/valid_sessions_rpe.pickle"
SPLITS_PATH = "/data/max_feat_kfold_splits.pickle"
SESS_BEHAVIOR_PATH = "/data/sub-SA_sess-{sess_name}_object_features.csv"
# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_{num_bins_smooth}_smooth.pickle"

# formatting will replace 'feedback_type' first, then replace others in another function
FEATURE_DIMS = ["Color", "Shape", "Pattern"]

MIN_BLOCKS_PER_RULE = 2
NUM_SPLITS = 5

def sub_select_trials(beh, frs):
    beh = behavioral_utils.filter_blocks_by_rule_occurence(beh, MIN_BLOCKS_PER_RULE)
    beh = behavioral_utils.filter_max_feat_correct(beh)
    frs = frs[frs.TrialNumber.isin(beh)]
    return beh, frs


def calc_and_save_session(sess_name, splits, model, norm_mode):
    start = time.time()
    print(f"Processing session {sess_name}")
    spikes_path = SESS_SPIKES_PATH
    beh, frs = io_utils.load_rpe_sess_beh_and_frs(
        sess_name, 
        beh_path=SESS_BEHAVIOR_PATH, 
        fr_path=spikes_path, 
        set_indices=False,
    )
    # get the values
    beh = behavioral_utils.get_feature_values_per_session(sess_name, beh)

    agg = frs.groupby(["UnitID", "TrialNumber"]).mean().reset_index()
    # hacky, but just pretend there's one timebin. 
    agg["TimeBins"] = 0
    mode = MODE
    agg = pd.merge(beh[["TrialNumber", "BlockNumber"]], agg, on="TrialNumber")
    if norm_mode == "zscore":
        # get behavior col, BlockNumber
        agg = spike_utils.zscore_frs(agg, group_cols=["UnitID", "BlockNumber"], mode=MODE)
        mode = f"Z{MODE}"
    elif norm_mode == "mean_sub": 
        agg = spike_utils.mean_sub_frs(agg, group_cols=["UnitID", "BlockNumber"], mode=MODE)
        mode = f"MeanSub{MODE}"

    beh, frs = sub_select_trials(beh, frs)
    # beh, frs = sub_select_trials(beh, frs)
    beh = beh.set_index(["TrialNumber"])
    agg = agg.set_index(["TrialNumber"])

    columns_to_flatten = ["MaxFeat"]
    input_columns = columns_to_flatten
    split_reses = []
    # for i in range(NUM_SPLITS):
    #     split_row = splits[(splits.session == sess_name) & (splits.split_idx == i)].iloc[0]
    #     print(f"Processing session {sess_name} split {i}, {len(split_row.train)} train, {len(split_row.test)} trials")
    #     split_res = glm_utils.fit_glm_for_data(
    #         (beh, agg), 
    #         input_columns=input_columns, 
    #         columns_to_flatten=columns_to_flatten,
    #         model_type=model,
    #         train_test_split=(split_row.train, split_row.test)
    #     )
    #     split_res["split_idx"] = i
    #     split_reses.append(split_res)
    # all_reses = pd.concat(split_reses)
    all_reses = glm_utils.fit_glm_for_data(
        (beh, agg), 
        input_columns=input_columns, 
        columns_to_flatten=columns_to_flatten,
        mode=mode,
        model_type=model,
    )
    all_reses.to_pickle(os.path.join(OUTPUT_DIR, f"{sess_name}_glm_{EVENT}_{mode}_{INTERVAL_SIZE}_{model}_maxfeat.pickle"))

    end = time.time()
    print(f"Session {sess_name} took {(end - start) / 60} minutes")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sess_idx', type=int, help="int from 0 - 27 denoting which session to run for")
    parser.add_argument('--model', type=str, default="LinearNoInt")
    parser.add_argument('--norm_mode', type=str, default=None)

    args = parser.parse_args()
    sess_idx = int(args.sess_idx)
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    splits = pd.read_pickle(SPLITS_PATH)
    print(f"There are {len(valid_sess)} sessions, processing row {sess_idx}")
    sess_name = valid_sess.iloc[sess_idx].session_name
    calc_and_save_session(sess_name, splits, args.model, args.norm_mode)

if __name__ == "__main__":
    main()

