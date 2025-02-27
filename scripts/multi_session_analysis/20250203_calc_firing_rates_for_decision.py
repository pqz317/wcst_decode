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

DECISION_MEDIAN = 600

PRE_INTERVAL = 0
POST_INTERVAL = DECISION_MEDIAN
INTERVAL_SIZE = 100
NUM_BINS_SMOOTH = 1
EVENT = "decision_warped"


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
    behavior_path = f"/data/patrick_res/behavior/{SUBJECT}/{sess_name}_object_features.csv"
    beh = pd.read_csv(behavior_path)
    valid_beh = beh[beh.Response.isin(["Correct", "Incorrect"])]
    spike_times = spike_general.get_spike_times(None, SUBJECT, sess_name, species_dir="/data")

    print("Calculating spikes by trial interval")
    interval_size_secs = INTERVAL_SIZE / 1000
    intervals = pd.DataFrame({
        "TrialNumber": valid_beh["TrialNumber"], 
        "IntervalStartTime": valid_beh["StimOnset"],
        "IntervalEndTime": valid_beh["FeedbackOnset"] - 800,
        "Duration": valid_beh["FeedbackOnset"] - 800 - valid_beh["StimOnset"]
    })
    intervals["scale"] = DECISION_MEDIAN / intervals["Duration"]

    spike_by_trial_interval = spike_utils.get_spikes_by_trial_interval(spike_times, intervals)
    spike_by_trial_interval["TrialNumber"] = spike_by_trial_interval["TrialNumber"].astype(int)
    spike_by_trial_interval = pd.merge(spike_by_trial_interval, intervals, on="TrialNumber", how="inner")
    spike_by_trial_interval["RelativeSpikeTime"] = spike_by_trial_interval.SpikeTime - spike_by_trial_interval.IntervalStartTime

    def warp_spike_times(spikes_in_trial):
        scale = spikes_in_trial.scale.iloc[0]
        spikes_in_trial["SpikeTime"] = spikes_in_trial.RelativeSpikeTime * scale + spikes_in_trial.IntervalStartTime
        return spikes_in_trial

    warped = spike_by_trial_interval.groupby("TrialNumber", group_keys=False).apply(warp_spike_times).reset_index(drop=True)

    all_units = spike_general.list_session_units(None, SUBJECT, sess_name, species_dir="/data")
    print(len(all_units))
    print("Calculating Firing Rates")
    end_bin = (PRE_INTERVAL + POST_INTERVAL) / 1000 + interval_size_secs
    print(warped.TrialNumber.nunique())
    print(warped.UnitID.nunique())
    print(valid_beh.TrialNumber.nunique())
    print(all_units.UnitID.nunique())

    firing_rates = spike_analysis.firing_rate(
        warped, 
        all_units, 
        bins=np.arange(0, end_bin, interval_size_secs), 
        smoothing=NUM_BINS_SMOOTH,
        trials=valid_beh.TrialNumber.unique()
    )
    firing_rates = pd.merge(firing_rates, intervals[["TrialNumber", "scale"]], on="TrialNumber")
    firing_rates["FiringRate"] = firing_rates.FiringRate / firing_rates.scale
    if not len(firing_rates.UnitID.unique()) == len(all_units.UnitID.unique()):
        raise ValueError(f"Session {sess_name}: {len(firing_rates.UnitID.unique())} units in firing rates when {len(all_units.UnitID.unique())} total")
    print("Saving")
    dir_path = f"/data/patrick_res/firing_rates/{SUBJECT}"
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    firing_rates.to_pickle(os.path.join(dir_path, f"{sess_name}_firing_rates_{PRE_INTERVAL}_{EVENT}_{POST_INTERVAL}_{INTERVAL_SIZE}_bins_{NUM_BINS_SMOOTH}_smooth.pickle"))

def main():
    valid_sess = pd.read_pickle(f"/data/patrick_res/sessions/{SUBJECT}/valid_sessions.pickle")
    valid_sess.apply(calc_firing_rate_for_interval, axis=1)

if __name__ == "__main__":
    main()