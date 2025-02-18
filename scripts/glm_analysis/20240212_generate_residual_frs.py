# per session, uses fitted GLMs to generate residual Firing Rates. 
# will do this for both spike counts and firing rates
# GLM will consist of a linear regression using features and RPE as separate set of variables

import argparse
import os
import numpy as np
import pandas as pd
import utils.behavioral_utils as behavioral_utils
import utils.information_utils as information_utils
import utils.io_utils as io_utils
import utils.glm_utils as glm_utils
from matplotlib import pyplot as plt
from multiprocessing import Pool
import time
from sklearn.linear_model import PoissonRegressor
from constants.glm_constants import *
from scipy.ndimage import gaussian_filter1d


SESSIONS_PATH = "/data/valid_sessions_rpe.pickle"
SESS_BEHAVIOR_PATH = "/data/sub-SA_sess-{sess_name}_object_features.csv"
# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_{num_bins_smooth}_smooth.pickle"
# RESIDUAL_SPIKES_PATH = "/data/{sess_name}_residual_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_{num_bins_smooth}_smooth.pickle"
RESIDUAL_SPIKES_PATH = "/data/{sess_name}_residual_feature_{feedback_type}_{interaction}{use_prev_trial_params}firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_{num_bins_smooth}_smooth.pickle"

FEATURE_DIMS = ["Color", "Shape", "Pattern"]

MODES = ["SpikeCounts", "FiringRate"]
MODEL = "Linear"

def resmooth_frs(frs):
    def smooth_unit_fr(unit_group):
        unit_sorted = unit_group.sort_values(by="TimeBins")
        unit_sorted["FiringRate"] = gaussian_filter1d(unit_sorted.SpikeCounts, NUM_BINS_SMOOTH)
        return unit_sorted
    frs.groupby(["UnitID", "TrialNumber"]).apply(smooth_unit_fr)
    return frs

def calc_and_save_session(sess_name, feedback_type, include_interactions, should_resmooth_frs, use_prev_trial_params):
    start = time.time()
    print(f"Processing session {sess_name}")
    data = io_utils.load_rpe_sess_beh_and_frs(
        sess_name, 
        beh_path=SESS_BEHAVIOR_PATH, 
        fr_path=SESS_SPIKES_PATH, 
        include_prev=use_prev_trial_params
    )

    input_cols = [feedback_type] + FEATURE_DIMS
    if include_interactions:
        # any interactions between feedback and features
        interaction_cols = [f"{dim}{feedback_type}" for dim in FEATURE_DIMS]
        input_cols = input_cols + interaction_cols
    if use_prev_trial_params:
        # use previous trial's fb, features, interactions instead of current
        input_cols = [f"Prev{col}" for col in input_cols]
    reses = []
    for mode in MODES: 
        res = glm_utils.fit_glm_for_data(data, input_cols, mode=mode, model_type=MODEL, include_predictions=True)
        res[mode] = res.actual - res.predicted
        reses.append(res[["UnitID", "TimeBins", "TrialNumber", mode]])
    merged = pd.merge(reses[0], reses[1], on=["UnitID", "TimeBins", "TrialNumber"])
    if should_resmooth_frs:
        merged = resmooth_frs(merged)
    residual_path = RESIDUAL_SPIKES_PATH.format(
        sess_name=sess_name, 
        feedback_type=feedback_type,
        interaction= "with_interaction_" if include_interactions else "",
        use_prev_trial_params = "prev_trial_params_" if use_prev_trial_params else "",
        pre_interval=PRE_INTERVAL, 
        event=EVENT, 
        post_interval=POST_INTERVAL, 
        interval_size=INTERVAL_SIZE,
        num_bins_smooth=NUM_BINS_SMOOTH,
    )
    merged.to_pickle(residual_path)
    end = time.time()
    print(f"Session {sess_name} took {(end - start) / 60} minutes")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sess_idx', type=int, help="int from 0 - 27 denoting which session to run for")
    parser.add_argument('--feedback_type', type=str, default="RPEGroup")
    parser.add_argument('--include_interactions', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--resmooth_frs', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--use_prev_trial_params', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    sess_idx = int(args.sess_idx)
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    print(f"There are {len(valid_sess)} sessions, processing row {sess_idx}")
    sess_name = valid_sess.iloc[sess_idx].session_name
    calc_and_save_session(sess_name, args.feedback_type, args.include_interactions, args.resmooth_frs, args.use_prev_trial_params)

if __name__ == "__main__":
    main()

