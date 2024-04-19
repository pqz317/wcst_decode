import argparse
import os
import numpy as np
import pandas as pd
import utils.behavioral_utils as behavioral_utils
import utils.information_utils as information_utils
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
RESIDUAL_SPIKES_PATH = "/data/{{sess_name}}_residual_feature_{feedback_type}_with_interaction_firing_rates_{{pre_interval}}_{{event}}_{{post_interval}}_{{interval_size}}_bins_{{num_bins_smooth}}_smooth.pickle"

FEATURE_DIMS = ["Color", "Shape", "Pattern"]
INTERACTIONS = [f"{dim}RPEGroup" for dim in FEATURE_DIMS]

def calc_and_save_session(sess_name, feedback_type, use_residual_fr):
    start = time.time()
    print(f"Processing session {sess_name}")
    if use_residual_fr:
        spikes_path = RESIDUAL_SPIKES_PATH.format(
            feedback_type=feedback_type,
        )
    else:
        spikes_path = SESS_SPIKES_PATH
    beh, frs = io_utils.load_rpe_sess_beh_and_frs(sess_name, beh_path=SESS_BEHAVIOR_PATH, fr_path=spikes_path, set_indices=False)
    # get the values
    beh = behavioral_utils.get_feature_values_per_session(sess_name, beh)
    beh = beh.set_index(["TrialNumber"])
    frs = frs.set_index(["TrialNumber"])

    value_cols = [feat + "Value" for feat in FEATURES]

    if not use_residual_fr:
        interaction_cols = [f"{dim}{feedback_type}" for dim in FEATURE_DIMS]
        columns_to_flatten = [feedback_type] + FEATURE_DIMS + interaction_cols
        input_columns = columns_to_flatten + value_cols
    else: 
        columns_to_flatten = []
        input_columns = value_cols
    value_reses = glm_utils.fit_glm_for_data((beh, frs), input_columns=input_columns, columns_to_flatten=columns_to_flatten)
    residual_str = "residual_fr" if use_residual_fr else "normal_fr"
    value_reses.to_pickle(os.path.join(OUTPUT_DIR, f"{sess_name}_glm_{feedback_type}_{residual_str}_{EVENT}_{MODE}_{INTERVAL_SIZE}_{MODEL}_values.pickle"))

    end = time.time()
    print(f"Session {sess_name} took {(end - start) / 60} minutes")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sess_idx', type=int, help="int from 0 - 27 denoting which session to run for")
    parser.add_argument('--feedback_type', type=str, default="RPEGroup")
    parser.add_argument('--use_residual_fr', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    sess_idx = int(args.sess_idx)
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    print(f"There are {len(valid_sess)} sessions, processing row {sess_idx}")
    sess_name = valid_sess.iloc[sess_idx].session_name
    calc_and_save_session(sess_name, args.feedback_type, args.use_residual_fr)

if __name__ == "__main__":
    main()

