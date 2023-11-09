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

species = 'nhp'
subject = 'SA'
exp = 'WCST'
session = 20180802  # this is the session for which there are spikes at the moment.    
pre_interval = 300
post_interval = 500

def calc_for_session(row):
    # grab behavioral data, spike data, trial numbers. 
    sess_name = row.session_name
    print(f"Processing session {sess_name}")
    print("Loading files")
    behavior_path = f"/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
    beh = pd.read_csv(behavior_path)

    valid_beh = behavioral_utils.get_valid_trials(beh)
    trial_numbers = np.unique(valid_beh.TrialNumber)
    spike_times = spike_general.get_spike_times(None, subject, session)

    raw_fixation_times = io_utils.get_raw_fixation_times(fs, subject, session)
    

    print("Calculating spikes by trial interval")

    fixation_features = behavioral_utils.get_fixation_features(behavior_data, raw_fixation_times)

    # fixation_features = fixation_features[fixation_features["TrialNumber"].isin(trial_numbers)]
    # valids = fixation_features.loc[~(fixation_features["ItemChosen"] == fixation_features["ItemNumber"])]

    first_fixations = behavioral_utils.get_first_fixations_for_cards(fixation_features)
    no_selected_fixations = behavioral_utils.remove_selected_fixation(first_fixations)
    valids = no_selected_fixations[no_selected_fixations["TrialNumber"].isin(trial_numbers)]
    print(f"Number of valid fixations: {len(valids.FixationNum.unique())}")

    # finds intervals aligned on fixation start
    intervals = pd.DataFrame(columns=["IntervalID", "IntervalStartTime", "IntervalEndTime"])
    intervals["IntervalID"] = valids["FixationNum"]
    intervals["IntervalStartTime"] = valids["FixationStart"] - pre_interval
    intervals["IntervalEndTime"] = valids["FixationStart"] + post_interval

    spike_by_trial_interval = spike_utils.get_spikes_by_interval(spike_times, intervals)
    print(f"Number of valid spike intervals: {len(spike_by_trial_interval.IntervalID.unique())}")
    end_bin = (pre_interval + post_interval) / 1000 + 0.1
    print("Calculating Firing Rates")
    firing_rates = spike_utils.get_firing_rates_by_interval(spike_by_trial_interval, bins=np.arange(0, end_bin, 0.1), smoothing=1)
    print(f"Number of valid firing rate intervals: {len(firing_rates.IntervalID.unique())}")

    print("Saving")
    firing_rates.to_pickle(fs.open(f"l2l.pqz317.scratch/firing_rates_{pre_interval}_filtered_fixationstart_{post_interval}.pickle", "wb"))
    spike_by_trial_interval.to_pickle(fs.open(f"l2l.pqz317.scratch/spike_by_trial_interval_{pre_interval}_filtered_fixationstart_{post_interval}.pickle", "wb"))

if __name__ == "__main__":
    main()