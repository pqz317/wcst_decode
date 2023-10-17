import os
import numpy as np
import pandas as pd
import utils.behavioral_utils as behavioral_utils
import utils.information_utils as information_utils
from matplotlib import pyplot as plt
from multiprocessing import Pool
import time

EVENT = "FeedbackOnset"  # event in behavior to align on
PRE_INTERVAL = 1300   # time in ms before event
POST_INTERVAL = 1500  # time in ms after event
INTERVAL_SIZE = 100  # size of interval in ms

# the output directory to store the data
OUTPUT_DIR = "/data/patrick_scratch/information"
SESSIONS_PATH = "/data/patrick_scratch/multi_sess/valid_sessions_rpe.pickle"

# path for each session, specifying behavior
SESS_BEHAVIOR_PATH = "/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/patrick_scratch/multi_sess/{sess_name}/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins.pickle"

DATA_MODE = "SpikeCounts"

FEATURE_DIMS = ["Color", "Shape", "Pattern"]
SHUFFLED_FEATURE_DIMS = ["ShuffledColor", "ShuffledShape", "ShuffledPattern"]

NUM_SHUFFLES = 1000
NUM_PROCESSES = 32

SEED = 42



def load_data(sess_name):
    behavior_path = SESS_BEHAVIOR_PATH.format(sess_name=sess_name)
    beh = pd.read_csv(behavior_path)

    # filter trials 
    valid_beh = behavioral_utils.get_valid_trials(beh)

    # grab the features of the selected card
    feature_selections = behavioral_utils.get_selection_features(valid_beh)
    valid_beh = pd.merge(valid_beh, feature_selections, on="TrialNumber", how="inner")
    valid_beh = valid_beh.set_index(["TrialNumber"])

    spikes_path = SESS_SPIKES_PATH.format(
        sess_name=sess_name, 
        pre_interval=PRE_INTERVAL, 
        event=EVENT, 
        post_interval=POST_INTERVAL, 
        interval_size=INTERVAL_SIZE
    )
    frs = pd.read_pickle(spikes_path)
    frs = frs.set_index(["TrialNumber"])
    return valid_beh, frs


def calc_for_shuffle(args):
    i, sess_name = args
    print(f"Calculating shuffle #{i}")
    valid_beh, frs = load_data(sess_name)
    rng = np.random.default_rng()
    for feature_dim in FEATURE_DIMS:
        labels = valid_beh[feature_dim].values
        rng.shuffle(labels)
        valid_beh[f"Shuffled{feature_dim}"] = labels
    shuffled_data = pd.merge(frs, valid_beh, on="TrialNumber")[["UnitID", "TimeBins", "SpikeCounts"] + SHUFFLED_FEATURE_DIMS]
    shuffled_data.set_index(["UnitID", "TimeBins"])
    shuffled_mi = information_utils.calc_mutual_information_per_unit_and_time(shuffled_data, "SpikeCounts", SHUFFLED_FEATURE_DIMS)
    shuffled_mi["ShuffleIdx"] = i
    return shuffled_mi


def calc_mi(session):
    valid_beh, frs = load_data(session)
    data = pd.merge(frs, valid_beh, on="TrialNumber")[["UnitID", "TimeBins", "SpikeCounts"] + FEATURE_DIMS]
    data.set_index(["UnitID", "TimeBins"])
    mi = information_utils.calc_mutual_information_per_unit_and_time(data, "SpikeCounts", FEATURE_DIMS)
    return mi

def create_null_shuffled(session):
    args = [(i, session) for i in range(NUM_SHUFFLES)]
    with Pool(processes=NUM_PROCESSES) as pool:
        res = pool.map(calc_for_shuffle, args)
    shuffled_mis = pd.concat(res)
    return shuffled_mis

def calc_null_stats(shuffled_mis):
    shuffled_mis.set_index(["UnitID", "TimeBins"])
    null_stats = shuffled_mis.groupby(["UnitID", "TimeBins"]).apply(
        lambda group: information_utils.calculate_shuffled_stats(group, FEATURE_DIMS)
    ).reset_index()
    return null_stats
    
def calc_and_save_session(session):
    start = time.time()
    print(f"Processing session {session}")
    mi = calc_mi(session)
    mi.to_pickle(os.path.join(OUTPUT_DIR, f"{session}_features_mi.pickle"))
    shuffled_mis = create_null_shuffled(session)
    shuffled_mis.to_pickle(os.path.join(OUTPUT_DIR, f"{session}_features_null_shuffled.pickle"))
    null_stats = calc_null_stats(shuffled_mis)
    null_stats.to_pickle(os.path.join(OUTPUT_DIR, f"{session}_features_null_stats.pickle"))
    end = time.time()
    print(f"Session {session} took {(end - start) / 60} minutes")


def main():
    start = time.time()
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    valid_sess.apply(lambda row: calc_and_save_session(row.session_name), axis=1)
    end = time.time()
    print(f"Whole script took {(end - start) / 60} minutes")


if __name__ == "__main__":
    main()

