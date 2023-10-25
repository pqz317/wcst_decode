import os
import numpy as np
import pandas as pd
import utils.behavioral_utils as behavioral_utils
import utils.information_utils as information_utils
from matplotlib import pyplot as plt
from multiprocessing import Pool
import time

# PRE_INTERVAL = 500
# POST_INTERVAL = 500
# INTERVAL_SIZE = 100
# NUM_BINS_SMOOTH = 1
# EVENT = "StimOnset"

PRE_INTERVAL = 1300
POST_INTERVAL = 1500
INTERVAL_SIZE = 100
NUM_BINS_SMOOTH = 1
EVENT = "FeedbackOnset"

# the output directory to store the data
OUTPUT_DIR = "/data/patrick_scratch/information"
SESSIONS_PATH = "/data/patrick_scratch/multi_sess/valid_sessions_rpe.pickle"

# path for each session, specifying behavior
SESS_BEHAVIOR_PATH = "/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/patrick_scratch/multi_sess/{sess_name}/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_{num_bins_smooth}_smooth.pickle"

DATA_MODE = "SpikeCounts"

NUM_SHUFFLES = 1000
NUM_PROCESSES = 32


MIN_BLOCKS_PER_RULE = 3

SEED = 42



def load_data(row, show_message=False):
    sess_name = row.session_name
    behavior_path = SESS_BEHAVIOR_PATH.format(sess_name=sess_name)
    beh = pd.read_csv(behavior_path)

    # filter trials 
    valid_beh = behavioral_utils.get_valid_trials(beh)
    num_blocks_per_rule = valid_beh.groupby("CurrentRule").BlockNumber.agg('nunique')
    valid_rules = num_blocks_per_rule[num_blocks_per_rule >= MIN_BLOCKS_PER_RULE].index
    has_enough = len(valid_rules) >= 2
    if show_message and has_enough:
        print(f"Session {sess_name} has {len(valid_rules)} rules with more than {MIN_BLOCKS_PER_RULE} blocks, using...")
    if show_message and not has_enough:
        print(f"Session {sess_name} has {len(valid_rules)} rules with more than {MIN_BLOCKS_PER_RULE} blocks, skipping session...")
    if not has_enough:
        return None

    valid_beh = valid_beh[valid_beh.CurrentRule.isin(valid_rules)]
    # use last N trials per block
    valid_beh = behavioral_utils.get_last_n_corrects_per_block(valid_beh, 8)

    spikes_path = SESS_SPIKES_PATH.format(
        sess_name=sess_name, 
        pre_interval=PRE_INTERVAL, 
        event=EVENT, 
        post_interval=POST_INTERVAL, 
        interval_size=INTERVAL_SIZE, 
        num_bins_smooth=NUM_BINS_SMOOTH
    )
    frs = pd.read_pickle(spikes_path)
    frs = frs.set_index(["TrialNumber"])
    return valid_beh, frs


def calc_for_shuffle(args):
    i, row = args
    print(f"Calculating shuffle #{i}")
    valid_beh, frs = load_data(row)
    rng = np.random.default_rng()
    def get_rule_of_block(block):
        rule = block.CurrentRule.unique()[0]
        return rule
    block_to_rule = valid_beh.groupby("BlockNumber").apply(get_rule_of_block).reset_index(name="CurrentBlockRule")
    labels = block_to_rule.CurrentBlockRule.values.copy()
    rng.shuffle(labels)
    block_to_rule["ShuffledCurrentRule"] = labels
    valid_beh = pd.merge(valid_beh, block_to_rule, on="BlockNumber")

    shuffled_data = pd.merge(frs, valid_beh, on="TrialNumber")[["UnitID", "TimeBins", "SpikeCounts", "ShuffledCurrentRule"]]
    shuffled_data.set_index(["UnitID", "TimeBins"])
    shuffled_mi = information_utils.calc_mutual_information_per_unit_and_time(shuffled_data, "SpikeCounts", ["ShuffledCurrentRule"])
    shuffled_mi["ShuffleIdx"] = i
    return shuffled_mi


def calc_mi(valid_beh, frs):
    data = pd.merge(frs, valid_beh, on="TrialNumber")[["UnitID", "TimeBins", "SpikeCounts", "CurrentRule"]]
    data.set_index(["UnitID", "TimeBins"])
    mi = information_utils.calc_mutual_information_per_unit_and_time(data, "SpikeCounts", ["CurrentRule"])
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
        lambda group: information_utils.calculate_shuffled_stats(group, ["CurrentRule"])
    ).reset_index()
    return null_stats
    
def calc_and_save_session(row):
    session = row.session_name
    start = time.time()
    print(f"Processing session {session}")
    res = load_data(row)
    if res is None:
        return
    valid_beh, frs = res
    mi = calc_mi( valid_beh, frs)
    mi.to_pickle(os.path.join(OUTPUT_DIR, f"{session}_rule_{EVENT}_mi.pickle"))

    shuffled_mis = create_null_shuffled(row)
    shuffled_mis.to_pickle(os.path.join(OUTPUT_DIR, f"{session}_rule_{EVENT}_null_shuffled.pickle"))

    null_stats = calc_null_stats(shuffled_mis)
    null_stats.to_pickle(os.path.join(OUTPUT_DIR, f"{session}_rule_{EVENT}_null_stats.pickle"))

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

