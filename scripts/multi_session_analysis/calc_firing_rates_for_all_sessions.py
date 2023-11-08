import numpy as np
import pandas as pd
from spike_tools import (
    general as spike_general,
    analysis as spike_analysis,
)
import utils.behavioral_utils as behavioral_utils
import utils.spike_utils as spike_utils
import os

"""
Creates firing rates dataframes for all sessions, saves them invdividually
Requires a sessions dataframe, with with column `session_name` that identifies each session

For each session, creates a firing_rates dataframe that should have columns: 
UnitID, TrialNumber, TimeBins, and some data columns, like SpikeCounts or FiringRates
"""

SPECIES = 'nhp'
SUBJECT = 'SA'

# PRE_INTERVAL = 150
# POST_INTERVAL = 350
# INTERVAL_SIZE = 100
# EVENT = "FixationOnCross"

PRE_INTERVAL = 500
POST_INTERVAL = 500
INTERVAL_SIZE = 100
NUM_BINS_SMOOTH = 1
EVENT = "StimOnset"

# PRE_INTERVAL = 1300
# POST_INTERVAL = 1500
# INTERVAL_SIZE = 50
# NUM_BINS_SMOOTH = 1
# EVENT = "FeedbackOnset"


def calc_firing_rate_for_interval(row):
    """
    For a session, per trial, aligns spike data to a specific behavioral event, 
    Bins spikes to specified bin sizes 
    Calculates spike counts, firing rates per bin
    Stores into a dataframe with columns:
            - UnitID
            - TrialNumber
            - TimeBins
            - SpikeCounts
            - FiringRate
    """
    sess_name = row.session_name
    print(f"Processing session {sess_name}")
    print("Loading files")
    behavior_path = f"/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
    beh = pd.read_csv(behavior_path)
    valid_beh = beh[beh.Response.isin(["Correct", "Incorrect"])]
    spike_times = spike_general.get_spike_times(None, SUBJECT, sess_name, species_dir="/data")

    print("Calculating spikes by trial interval")
    interval_size_secs = INTERVAL_SIZE / 1000
    intervals = behavioral_utils.get_trial_intervals(valid_beh, EVENT, PRE_INTERVAL, POST_INTERVAL)
    
    spike_by_trial_interval = spike_utils.get_spikes_by_trial_interval(spike_times, intervals)
    end_bin = (PRE_INTERVAL + POST_INTERVAL) / 1000 + interval_size_secs

    print("Calculating Firing Rates")
    firing_rates = spike_analysis.firing_rate(spike_by_trial_interval, spike_by_trial_interval, bins=np.arange(0, end_bin, interval_size_secs), smoothing=NUM_BINS_SMOOTH)

    print("Saving")
    dir_path = f"/data/patrick_res/multi_sess/{sess_name}"
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    firing_rates.to_pickle(os.path.join(dir_path, f"{sess_name}_firing_rates_{PRE_INTERVAL}_{EVENT}_{POST_INTERVAL}_{INTERVAL_SIZE}_bins_{NUM_BINS_SMOOTH}_smooth.pickle"))


def main():
    valid_sess = pd.read_pickle("/data/patrick_res/multi_sess/valid_sessions.pickle")
    valid_sess.apply(calc_firing_rate_for_interval, axis=1)

if __name__ == "__main__":
    main()