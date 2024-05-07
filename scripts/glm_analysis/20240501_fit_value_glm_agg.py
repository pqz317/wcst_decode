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
SESS_BEHAVIOR_PATH = "/data/sub-SA_sess-{sess_name}_object_features.csv"
# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_{num_bins_smooth}_smooth.pickle"

# formatting will replace 'feedback_type' first, then replace others in another function
FEATURE_DIMS = ["Color", "Shape", "Pattern"]

def calc_and_save_session(sess_name, model, block_zscore_fr):
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

    beh = beh.set_index(["TrialNumber"])
    agg = frs.groupby(["UnitID", "TrialNumber"]).mean().reset_index()
    # hacky, but just pretend there's one timebin. 
    agg["TimeBins"] = 0
    mode = MODE
    if block_zscore_fr:
        # get behavior col, BlockNumber
        agg = pd.merge(beh, agg, on="TrialNumber")
        agg = spike_utils.zscore_frs(agg, group_cols=["UnitID", "BlockNumber"], mode=MODE)
        mode = f"Z{MODE}"
    agg = agg.set_index(["TrialNumber"])


    value_cols = [feat + "Value" for feat in FEATURES]  
    columns_to_flatten = []
    input_columns = value_cols
    value_reses = glm_utils.fit_glm_for_data((beh, agg), input_columns=input_columns, columns_to_flatten=columns_to_flatten)

    value_reses.to_pickle(os.path.join(OUTPUT_DIR, f"{sess_name}_glm_{EVENT}_{mode}_{INTERVAL_SIZE}_{model}_values_agg.pickle"))

    end = time.time()
    print(f"Session {sess_name} took {(end - start) / 60} minutes")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sess_idx', type=int, help="int from 0 - 27 denoting which session to run for")
    parser.add_argument('--model', type=str, default="Linear")
    parser.add_argument('--block_zscore_fr', action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()
    sess_idx = int(args.sess_idx)
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    print(f"There are {len(valid_sess)} sessions, processing row {sess_idx}")
    sess_name = valid_sess.iloc[sess_idx].session_name
    calc_and_save_session(sess_name, args.model, args.block_zscore_fr)

if __name__ == "__main__":
    main()

