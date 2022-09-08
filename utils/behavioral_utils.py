import numpy as np
import pandas as pd

# sorted features by shape, color, pattern

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
    intervals_df = pd.DataFrame(columns=["TrialNumber", "IntervalStartTime", "IntervalEndTime"])
    intervals_df["TrialNumber"] = trial_event_times["TrialNumber"].astype(int)
    intervals_df["IntervalStartTime"] = trial_event_times[event] - pre_interval
    intervals_df["IntervalEndTime"] = trial_event_times[event] + post_interval
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

def get_shuffled_card_idxs(behavioral_data, seed=42):
    """Per Trial, shuffles the indexes of the cards, so that the correct card
    is no longer at index 0. 
    """
    rng = np.random.default_rng(seed)
    data = []

    for _, row in behavioral_data.iterrows():
        new_row = {}
        shuffled_idxs = np.arange(0, 4)
        rng.shuffle(shuffled_idxs)
        for feature_dim in ["Color", "Shape", "Pattern"]:
            for old_idx, new_idx in enumerate(shuffled_idxs):
                new_row[f"Item{new_idx}{feature_dim}"] = row[f"Item{old_idx}{feature_dim}"]
        new_row["ItemChosen"] = shuffled_idxs[int(row["ItemChosen"])]
        new_row["TrialNumber"] = row["TrialNumber"]
        data.append(new_row)
    return pd.DataFrame(data)

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
    fixation_features = pd.DataFrame(fixation_features)
    # add a new column that's a numbering of rows, used as an ID later. 
    fixation_features["FixationNum"] = fixation_features.reset_index().index
    return fixation_features


def get_first_fixations_for_cards(fixation_features):
    """In fixation_features, consecutive fixations onto the same card 
    get logged as separate fixations. in these cases, only subselect the
    first fixation per card. 

    Args:
        fixation_features: df with TrialNumber, ItemNumber, FixationNum
    Returns:
        A filtered fixation_features where only the first fixation within a 
        consective series of fixations for a card is included
    """
    df = fixation_features.copy()
    df["PrevItemNumber"] = df["ItemNumber"].shift()
    df["PrevTrialNumber"] = df["TrialNumber"].shift()
    df["SameCardFixation"] = (~( 
        (df["ItemNumber"] == df["PrevItemNumber"]) & 
        (df["PrevTrialNumber"] == df["TrialNumber"]) 
    )).cumsum()
    same_card_grouped = df.groupby(["SameCardFixation"], as_index=False)
    first_fixations = same_card_grouped.apply(lambda x: x.iloc[0])
    return first_fixations


def remove_selected_fixation(first_fixations_for_cards):
    """For a df of first fixations of cards, removes the fixation deemed as the selection. 

    Args:
        first_fixations_for_cards: df with TrialNumber, ItemNumber, FixationNum, 
            first fixation for a card of consecutive fixations
    
    Returns:
        A filtered first_fixations_for_cards where the fixation deemed as a selection
        is removed
    """
    def removed_selected_card(fixated_cards_per_trial):
        # finds last card in this trial
        last_card = fixated_cards_per_trial.iloc[-1]
        if last_card.ItemChosen == last_card.ItemNumber:
            # last card fixated on is indeed selected card
            return fixated_cards_per_trial[0:-1]
        else:
            return fixated_cards_per_trial

    trials_grouped = first_fixations_for_cards.groupby(["TrialNumber"], as_index=False)
    return trials_grouped.apply(removed_selected_card).reset_index()


def exclude_first_block(behavioral_data):
    return behavioral_data[behavioral_data.BlockNumber != 0]


def get_figured_out_trials(behavioral_data):
    """
    Get trials that fall into the 8/8 or 16/20 conditions
    for rule switches
    """
    def label_trials(block_group):
        block_len = len(block_group)
        block_group["TrialUntilRuleChange"] = block_len - block_group["TrialAfterRuleChange"]
        last_eight = block_group[block_group["TrialUntilRuleChange"] <= 8]
        if (last_eight.Response == "Correct").all():
            return last_eight
        else:
            return block_group[block_group["TrialUntilRuleChange"] <= 20]
    block_groups = behavioral_data.groupby(["BlockNumber"], as_index=False)
    return block_groups.apply(label_trials).reset_index()


def get_not_figured_out_trials(behavioral_data):
    figured_out_trial_nums = get_figured_out_trials(behavioral_data).TrialNumber
    return behavioral_data[~behavioral_data.TrialNumber.isin(figured_out_trial_nums)]


def get_first_n_in_blocks(behavioral_data):
    pass

def get_last_n_in_blocks(behavioral_data):
    pass