import numpy as np
import pandas as pd

def get_trial_intervals(behavioral_data, event="FeedbackOnset", pre_interval=0, post_interval=0):
    """
    Per trial, finds time interval surrounding some event in the behavioral data
    Returns: DataFrame with num_trials length, columns: TrialNumber, IntervalStartTime, IntervalEndTime
    """
    trial_event_times = behavioral_data[["TrialNumber", event]]

    intervals = np.empty((len(trial_event_times), 3))
    intervals[:, 0] = trial_event_times["TrialNumber"]
    intervals[:, 1] = trial_event_times[event] - pre_interval
    intervals[:, 2] = trial_event_times[event] + post_interval
    intervals_df = pd.DataFrame(intervals, columns=["TrialNumber", "IntervalStartTime", "IntervalEndTime"])
    return intervals_df