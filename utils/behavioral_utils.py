import numpy as np
import pandas as pd

def get_trial_intervals(behavioral_data, event="FeedbackOnset", pre_interval=0, post_interval=0):
    """Per trial, finds time interval surrounding some event in the behavioral data

    Args:
        behavioral_data: Dataframe describing each trial, must contain
            columns: TrialNumber, whatever 'event' param describes
        event: name of event to align around, must be present as a
            column name in behavioral_data Dataframe
        pre_interval: number of miliseconds before event
        post_interval: number of miliseconds after event

    Returns:
        DataFrame with num_trials length, columns: TrialNumber,
        IntervalStartTime, IntervalEndTime
    """
    trial_event_times = behavioral_data[["TrialNumber", event]]

    intervals = np.empty((len(trial_event_times), 3))
    intervals[:, 0] = trial_event_times["TrialNumber"]
    intervals[:, 1] = trial_event_times[event] - pre_interval
    intervals[:, 2] = trial_event_times[event] + post_interval
    intervals_df = pd.DataFrame(intervals, columns=["TrialNumber", "IntervalStartTime", "IntervalEndTime"])
    return intervals_df

def get_selection_features(behavioral_data):
    """Per trial, gets the Color, Shape Pattern of the selected card

    Args:
        behavioral_data: Dataframe describing each trial, must contain
            columns: - ItemChosen, either 0, 1, 2, 3, index of item that
            was chosen that trial - For each idx 0, 1, 2, 3, columns
            'Item<idx><feature>', where feature is 'Color', 'Pattern' or
            'Shape'

    Returns:
        DataFrame with num_trials length, columns: TrialNumber, Color,
        Pattern, Shape
    """
    selections = []
    for _, row in behavioral_data.iterrows():
        item_chosen = int(row["ItemChosen"])
        color = row[f"Item{item_chosen}Color"]
        shape = row[f"Item{item_chosen}Shape"]
        pattern = row[f"Item{item_chosen}Pattern"]
        selections.append([row["TrialNumber"], color, shape, pattern])
    return pd.DataFrame(selections, columns=["TrialNumber", "Color", "Shape", "Pattern"])


def get_fixation_features(behavioral_data, raw_fixation_times):
    """Given behavioral data and fixation times, constructs table
    with fixation times and features of card fixating on. 

    Args:
        behavioral_data: dataframe of num_trials length, containing 
            Item{0/1/2/3}{Shape/Color/Pattern} columns, and ItemChosen column
        raw_fixation_times: np.array of num trials length, with each element 
            as a separate np array describing every fixation during the trial 

    Returns:
        dataframe where each row is a card fixation with columns:
            TrialNumber, ItemChosen, ItemNumber, Shape, Color, Pattern, FixationStart, FixationEnd
    """
    fixation_features = []
    for trial_idx, trial_fixations in enumerate(raw_fixation_times):
        # trial numbers indexed by 1
        trial_num = trial_idx + 1
        if len(trial_fixations) == 0:
            # trial's empty, skip
            continue
        for fixation in trial_fixations[0]:
            if len(fixation) == 0:
                # fixation's empty, skip
                continue
            item_idx = int(fixation[0])
            fixation_start = fixation[1]
            fixation_end = fixation[2]
            trial_beh = behavioral_data[behavioral_data["TrialNumber"] == trial_num]
            item_chosen = np.squeeze(trial_beh["ItemChosen"])
            color = np.squeeze(trial_beh[f"Item{item_idx}Color"])
            shape = np.squeeze(trial_beh[f"Item{item_idx}Shape"])
            pattern = np.squeeze(trial_beh[f"Item{item_idx}Pattern"])
            fixation_features.append({
                "TrialNumber": trial_num, 
                "ItemChosen": item_chosen,
                "ItemNumber": item_idx, 
                "Color": color, 
                "Shape": shape, 
                "Pattern": pattern, 
                "FixationStart": fixation_start, 
                "FixationEnd": fixation_end
            })
    return pd.DataFrame(fixation_features)
