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

def calc_and_save_session(sess_name):
    start = time.time()
    print(f"Processing session {sess_name}")
    data = io_utils.load_rpe_sess_beh_and_frs(sess_name, beh_path=SESS_BEHAVIOR_PATH, fr_path=SESS_SPIKES_PATH)

    interaction_input_cols = ["RPEGroup"] + FEATURE_DIMS + INTERACTIONS
    interaction_reses = glm_utils.fit_glm_for_data(data, interaction_input_cols)
    interaction_reses.to_pickle(os.path.join(OUTPUT_DIR, f"{sess_name}_glm_feature_rpe_interaction.pickle"))

    end = time.time()
    print(f"Session {sess_name} took {(end - start) / 60} minutes")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sess_idx', type=int, help="int from 0 - 27 denoting which session to run for")
    args = parser.parse_args()
    sess_idx = int(args.sess_idx)
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    print(f"There are {len(valid_sess)} sessions, processing row {sess_idx}")
    sess_name = valid_sess.iloc[sess_idx].session_name
    calc_and_save_session(sess_name)

if __name__ == "__main__":
    main()

