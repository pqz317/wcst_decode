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

# the output directory to store the data
OUTPUT_DIR = "/data/res"
SESSIONS_PATH = "/data/valid_sessions_rpe.pickle"
SESS_BEHAVIOR_PATH = "/data/sub-SA_sess-{sess_name}_object_features.csv"
# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_{num_bins_smooth}_smooth.pickle"

FEATURE_DIMS = ["Color", "Shape", "Pattern"]
INTERACTIONS = [f"{dim}RPE" for dim in FEATURE_DIMS]

NUM_SHUFFLES = 1000

MODE = "FiringRate"

def calc_and_save_session(sess_name, shuffle_idx):
    start = time.time()
    print(f"Processing session {sess_name} shuffle {shuffle_idx}")
    beh, frs = io_utils.load_rpe_sess_beh_and_frs(sess_name, beh_path=SESS_BEHAVIOR_PATH, fr_path=SESS_SPIKES_PATH)
    rng = np.random.default_rng(shuffle_idx)

    input_columns = ["RPEGroup"] + FEATURE_DIMS + INTERACTIONS
    beh_inputs_to_shuffle = beh[input_columns]
    shuffle_columns = INTERACTIONS
    shuffled_beh = glm_utils.create_shuffles(beh_inputs_to_shuffle, shuffle_columns, rng)

    shuffled_res = glm_utils.fit_glm_for_data((shuffled_beh, frs), input_columns, mode=MODE)

    shuffled_res.to_pickle(os.path.join(OUTPUT_DIR, f"{sess_name}_glm_feature_rpe_shuffle_{shuffle_idx}.pickle"))
    end = time.time()
    print(f"Session {sess_name} took {(end - start) / 60} minutes")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('shuffle_idx', type=int, help="int from 0 - 999 denoting which session to run for")
    parser.add_argument('spike_mode', type=str, default="SpikeCounts")

    args = parser.parse_args()
    shuffle_idx = int(args.shuffle_idx)
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    print(f"processing shuffle index {shuffle_idx} for all {len(valid_sess)} sessions")
    valid_sess.apply(lambda row: calc_and_save_session(row.session_name, shuffle_idx), axis=1)

if __name__ == "__main__":
    main()

