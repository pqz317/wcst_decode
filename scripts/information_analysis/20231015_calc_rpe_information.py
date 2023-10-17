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

NUM_SHUFFLES = 1000
NUM_PROCESSES = 32

SEED = 42



def load_data(row):
    sess_name = row.session_name
    behavior_path = SESS_BEHAVIOR_PATH.format(sess_name=sess_name)
    beh = pd.read_csv(behavior_path)

    # filter trials 
    valid_beh = behavioral_utils.get_valid_trials(beh)

    # grab the features of the selected card
    valid_beh_rpes = behavioral_utils.get_rpes_per_session(sess_name, valid_beh)
    assert len(valid_beh) == len(valid_beh_rpes)
    pos_med = row.FE_pos_median
    neg_med = row.FE_neg_median
    # add median labels to 
    def add_group(row):
        rpe = row.RPE_FE
        group = None
        if rpe < neg_med:
            group = "more neg"
        elif rpe >= neg_med and rpe < 0:
            group = "less neg"
        elif rpe >= 0 and rpe < pos_med:
            group = "less pos"
        elif rpe > pos_med:
            group = "more pos"
        row["RPEGroup"] = group
        return row
    valid_beh_rpes = valid_beh_rpes.apply(add_group, axis=1)
    valid_beh_rpes = valid_beh_rpes.set_index(["TrialNumber"])

    spikes_path = SESS_SPIKES_PATH.format(
        sess_name=sess_name, 
        pre_interval=PRE_INTERVAL, 
        event=EVENT, 
        post_interval=POST_INTERVAL, 
        interval_size=INTERVAL_SIZE
    )
    frs = pd.read_pickle(spikes_path)
    frs = frs.set_index(["TrialNumber"])
    return valid_beh_rpes, frs


def calc_for_shuffle(args):
    i, row = args
    print(f"Calculating shuffle #{i}")
    valid_beh, frs = load_data(row)
    rng = np.random.default_rng()
    labels = valid_beh["RPEGroup"].values
    rng.shuffle(labels)
    valid_beh[f"ShuffledRPEGroup"] = labels
    shuffled_data = pd.merge(frs, valid_beh, on="TrialNumber")[["UnitID", "TimeBins", "SpikeCounts", "ShuffledRPEGroup"]]
    shuffled_data.set_index(["UnitID", "TimeBins"])
    shuffled_mi = information_utils.calc_mutual_information_per_unit_and_time(shuffled_data, "SpikeCounts", ["ShuffledRPEGroup"])
    shuffled_mi["ShuffleIdx"] = i
    return shuffled_mi


def calc_mi(row):
    valid_beh, frs = load_data(row)
    data = pd.merge(frs, valid_beh, on="TrialNumber")[["UnitID", "TimeBins", "SpikeCounts", "RPEGroup"]]
    data.set_index(["UnitID", "TimeBins"])
    mi = information_utils.calc_mutual_information_per_unit_and_time(data, "SpikeCounts", ["RPEGroup"])
    return mi

def create_null_shuffled(row):
    args = [(i, row) for i in range(NUM_SHUFFLES)]
    with Pool(processes=NUM_PROCESSES) as pool:
        res = pool.map(calc_for_shuffle, args)
    shuffled_mis = pd.concat(res)
    return shuffled_mis

def calc_null_stats(shuffled_mis):
    shuffled_mis.set_index(["UnitID", "TimeBins"])
    null_stats = shuffled_mis.groupby(["UnitID", "TimeBins"]).apply(
        lambda group: information_utils.calculate_shuffled_stats(group, ["RPEGroup"])
    ).reset_index()
    return null_stats
    
def calc_and_save_session(row):
    session = row.session_name
    start = time.time()
    print(f"Processing session {session}")
    mi = calc_mi(row)
    mi.to_pickle(os.path.join(OUTPUT_DIR, f"{session}_rpe_mi.pickle"))

    shuffled_mis = create_null_shuffled(row)
    shuffled_mis.to_pickle(os.path.join(OUTPUT_DIR, f"{session}_rpe_null_shuffled.pickle"))

    null_stats = calc_null_stats(shuffled_mis)
    null_stats.to_pickle(os.path.join(OUTPUT_DIR, f"{session}_rpe_null_stats.pickle"))

    end = time.time()
    print(f"Session {session} took {(end - start) / 60} minutes")


def main():
    start = time.time()
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    valid_sess.apply(calc_and_save_session, axis=1)
    end = time.time()
    print(f"Whole script took {(end - start) / 60} minutes")


if __name__ == "__main__":
    main()

