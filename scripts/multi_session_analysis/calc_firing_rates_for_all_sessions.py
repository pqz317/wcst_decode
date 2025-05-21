import numpy as np
import pandas as pd
from spike_tools import (
    general as spike_general,
    analysis as spike_analysis,
)
import utils.behavioral_utils as behavioral_utils
import utils.spike_utils as spike_utils
import os
from distutils.util import strtobool

import argparse


"""
Creates firing rates dataframes for all sessions, saves them invdividually
Requires a sessions dataframe, with with column `session_name` that identifies each session

For each session, creates a firing_rates dataframe that should have columns: 
UnitID, TrialNumber, TimeBins, and some data columns, like SpikeCounts or FiringRates
"""

SPECIES = 'nhp'
# SUBJECT = 'SA'

# PRE_INTERVAL = 500
# POST_INTERVAL = 500
# INTERVAL_SIZE = 50
# NUM_BINS_SMOOTH = 1
# EVENT = "FixationOnCross"

# PRE_INTERVAL = 1000
# POST_INTERVAL = 1000
# INTERVAL_SIZE = 100
# NUM_BINS_SMOOTH = 1
# EVENT = "StimOnset"

# PRE_INTERVAL = 1300
# POST_INTERVAL = 1500
# INTERVAL_SIZE = 100
# NUM_BINS_SMOOTH = 1
# EVENT = "FeedbackOnset"

SESSIONS_PATH = "/data/patrick_res/sessions/{sub}/valid_sessions.pickle"


def calc_firing_rate_for_interval(row, args):
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
    behavior_path = f"/data/patrick_res/behavior/{args.subject}/{sess_name}_object_features.csv"
    beh = pd.read_csv(behavior_path)
    valid_beh = beh[beh.Response.isin(["Correct", "Incorrect"])]
    spike_times = spike_general.get_spike_times(None, args.subject, sess_name, species_dir="/data")
    if spike_times is None: 
        print(f"No spikes for session {sess_name} detected, skipping")
        return None

    print("Calculating spikes by trial interval")
    interval_size_secs = args.interval_size / 1000
    intervals = behavioral_utils.get_trial_intervals(valid_beh, args.event, args.pre_interval, args.post_interval)
    
    spike_by_trial_interval = spike_utils.get_spikes_by_trial_interval(spike_times, intervals)
    end_bin = (args.pre_interval + args.post_interval) / 1000 + interval_size_secs

    all_units = spike_general.list_session_units(None, args.subject, sess_name, species_dir="/data")
    print(len(all_units))
    print("Calculating Firing Rates")
    firing_rates = spike_analysis.firing_rate(
        spike_by_trial_interval, 
        all_units, 
        bins=np.arange(0, end_bin, interval_size_secs), 
        smoothing=args.num_bins_smooth,
        trials=valid_beh.TrialNumber.unique()
    )
    if not len(firing_rates.UnitID.unique()) == len(all_units.UnitID.unique()):
        raise ValueError(f"Session {sess_name}: {len(firing_rates.UnitID.unique())} units in firing rates when {len(all_units.UnitID.unique())} total")
    print("Saving")
    dir_path = os.path.join(args.base_output_path, args.subject)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    full_file_name = os.path.join(dir_path, f"{sess_name}_firing_rates_{args.pre_interval}_{args.event}_{args.post_interval}_{args.interval_size}_bins_{args.num_bins_smooth}_smooth.pickle")
    print(f"For sub {args.subject}, session {sess_name}, storing FR of {firing_rates.UnitID.nunique()} units to {full_file_name}")
    if not args.dry_run: 
        firing_rates.to_pickle(full_file_name)
    return full_file_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', default="SA", type=str)
    parser.add_argument('--event', default="FeedbackOnset", type=str)
    parser.add_argument('--pre_interval', default=1300, type=int)
    parser.add_argument('--post_interval', default=1500, type=int)
    parser.add_argument('--interval_size', default=100, type=int)
    parser.add_argument('--num_bins_smooth', default=1, type=int)
    parser.add_argument('--base_output_path', default="/data/patrick_res/firing_rates/", type=str)
    parser.add_argument('--dry_run', default=True, type=lambda x: bool(strtobool(x)))
    args = parser.parse_args()

    print(f"Running in dry run: {args.dry_run}")
    valid_sess = pd.read_pickle(SESSIONS_PATH.format(sub=args.subject))
    print(f"processing {len(valid_sess)} sessions for {args.subject}")
    res = valid_sess.apply(lambda row: calc_firing_rate_for_interval(row, args), axis=1)
    print(f"generated frs for {len(res[~res.isna()])} sessions")

if __name__ == "__main__":
    main()