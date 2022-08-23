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

species = 'nhp'
subject = 'SA'
exp = 'WCST'
session = 20180802  # this is the session for which there are spikes at the moment.    
pre_interval = 300
post_interval = 500
interval_size = 100


def main():
    # grab behavioral data, spike data, trial numbers. 
    fs = s3fs.S3FileSystem()
    print("Grabbing stuff from s3")
    behavior_file = spike_general.get_behavior_path(subject, session)
    behavior_data = pd.read_csv(fs.open(behavior_file))
    valid_beh = behavior_data[behavior_data.Response.isin(["Correct", "Incorrect"])]
    spike_times = spike_general.get_spike_times(fs, subject, session)

    print("Calculating spikes by trial interval")
    interval_size_secs = interval_size / 1000
    intervals = behavioral_utils.get_trial_intervals(valid_beh, "FixationOnCross", pre_interval, post_interval)
    
    spike_by_trial_interval = spike_utils.get_spikes_by_trial_interval(spike_times, intervals)
    end_bin = (pre_interval + post_interval) / 1000 + interval_size_secs

    print("Calculating Firing Rates")
    firing_rates = spike_analysis.firing_rate(spike_by_trial_interval, bins=np.arange(0, end_bin, interval_size_secs), smoothing=1)

    print("Saving")
    firing_rates.to_pickle(fs.open(f"l2l.pqz317.scratch/firing_rates_{pre_interval}_crossfixation_{post_interval}_{interval_size}_bins.pickle", "wb"))
    spike_by_trial_interval.to_pickle(fs.open(f"l2l.pqz317.scratch/spike_by_trial_interval_{pre_interval}_crossfixation_{post_interval}_{interval_size}_bins.pickle", "wb"))

if __name__ == "__main__":
    main()