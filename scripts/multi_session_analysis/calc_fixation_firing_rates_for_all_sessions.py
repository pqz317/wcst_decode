# creates spike by trials, firing rates data for a specific interval

import numpy as np
import pandas as pd
from spike_tools import (
    general as spike_general,
    analysis as spike_analysis,
)
import s3fs
import utils.behavioral_utils as behavioral_utils
import utils.spike_utils as spike_utils
import utils.io_utils as io_utils

from lfp_tools import startup_local

import os

FEATURE_DIMS = ["Color", "Shape", "Pattern"]


PRE_INTERVAL = 300
POST_INTERVAL = 500
INTERVAL_SIZE = 100
NUM_BINS_SMOOTH = 1

SPECIES = 'nhp'
SUBJECT = 'SA'
TASK = "WCST"

def calc_for_session(row):
    # grab behavioral data, spike data, trial numbers. 
    sess_name = row.session_name
    print(f"Processing session {sess_name}")
    print("Loading files")
    behavior_path = f"/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
    beh = pd.read_csv(behavior_path)
    valid_beh = behavioral_utils.get_valid_trials(beh)

    res = startup_local.get_sac_dataframe(SPECIES, SUBJECT, TASK, sess_name, False)
    res = res[res.trial.isin(valid_beh.TrialNumber)]
    # saccade was onto something different
    other_cards = ["h1", "h2", "h3", "s1", "s2", "s3"]
    res = res[res.obj_end.isin(other_cards)]
    # saccade was onto something different
    res = res[res.obj_start != res.obj_end]
    merged = pd.merge(res, valid_beh, left_on="trial", right_on="TrialNumber")
    def get_features_from_card(row): 
        card_idx = int(row["obj_end"][1])
        for feature_dim in FEATURE_DIMS:
            row[feature_dim] = row[f"Item{card_idx}{feature_dim}"]
        return row
    merged = merged.apply(get_features_from_card, axis=1)
    # NOTE: Hack, this is not a trial number in reality, but everywhere else uses TrialNumber convention
    merged["TrialNumber"] = np.arange(len(merged))


    # finds intervals aligned on fixation start
    intervals = pd.DataFrame(columns=["IntervalID", "IntervalStartTime", "IntervalEndTime"])
    intervals["IntervalID"] = merged["TrialNumber"]
    intervals["IntervalStartTime"] = merged["time_end"] - PRE_INTERVAL
    intervals["IntervalEndTime"] = merged["time_end"] + POST_INTERVAL

    print("Calculating spikes by trial interval")
    interval_size_secs = INTERVAL_SIZE / 1000
    spike_times = spike_general.get_spike_times(None, SUBJECT, sess_name, species_dir="/data")
    spike_by_trial_interval = spike_utils.get_spikes_by_trial_interval(spike_times, intervals)
    end_bin = (PRE_INTERVAL + POST_INTERVAL) / 1000 + interval_size_secs

    print("Calculating Firing Rates")
    all_units = spike_general.list_session_units(None, SUBJECT, sess_name, species_dir="/data")
    firing_rates = spike_analysis.firing_rate(
        spike_by_trial_interval, 
        all_units, 
        bins=np.arange(0, end_bin, interval_size_secs), 
        smoothing=NUM_BINS_SMOOTH, 
        trials=merged.TrialNumber.unique()
    )
    print("Saving")
    dir_path = f"/data/patrick_res/firing_rates"
    firing_rates.to_pickle(os.path.join(dir_path, f"{sess_name}_firing_rates_{PRE_INTERVAL}_fixation_{POST_INTERVAL}_{INTERVAL_SIZE}_bins_{NUM_BINS_SMOOTH}_smooth.pickle"))
    merged.to_pickle(os.path.join(dir_path, f"{sess_name}_fixations.pickle"))

if __name__ == "__main__":
    valid_sess = pd.read_pickle("/data/sessions/valid_sessions_rpe.pickle")
    valid_sess.apply(calc_for_session, axis=1)