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

def get_selection_features(behavioral_data):
    """
    Per trial, gets the Color, Shape Pattern of the selected card
    Returns: DataFrame with TrialNumber, Color, Pattern, Shape columns
    """
    # columns: TrialNumber, Color, Shape, Pattern
    selections = []
    for _, row in behavioral_data.iterrows():
        item_chosen = int(row["ItemChosen"])
        color = row[f"Item{item_chosen}Color"]
        shape = row[f"Item{item_chosen}Shape"]
        pattern = row[f"Item{item_chosen}Pattern"]
        selections.append([row["TrialNumber"], color, shape, pattern])
    return pd.DataFrame(selections, columns=["TrialNumber", "Color", "Shape", "Pattern"])
