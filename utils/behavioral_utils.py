import numpy as np
import pandas as pd
from lfp_tools import startup

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

def get_first_correct_in_blocks(behavioral_data):
    def first_correct(block_group):
        corrects = block_group[block_group.Response == "Correct"]
        if len(corrects) > 0: 
            return corrects.iloc[0]
        else: 
            return None
    # filter out last block
    last_block = behavioral_data.BlockNumber.max()
    without_last_block = behavioral_data[behavioral_data.BlockNumber != last_block]
    block_grouped = without_last_block.groupby(["BlockNumber"], as_index=False)
    first_corrects = block_grouped.apply(first_correct).dropna(how='all').reset_index()
    return first_corrects

def get_trial_after_first_correct_stats(behavioral_data, first_corrects):
    trial_nums = behavioral_data.TrialNumber.values
    first_correct_idxs = np.argwhere(np.isin(trial_nums, first_corrects.TrialNumber.values)).squeeze()
    next_idxs = first_correct_idxs + 1
    next_trial_nums = trial_nums[next_idxs]
    next_trials = behavioral_data[behavioral_data.TrialNumber.isin(next_trial_nums)]
    next_trials["LastPattern"] = first_corrects.Item0Pattern.values
    next_trials["LastColor"] = first_corrects.Item0Color.values
    next_trials["LastShape"] = first_corrects.Item0Shape.values

    def get_num_prev_features(row):
        has_two_idx = -1
        has_one_idx = -1
        for i in range(4):
            num_prev = 0
            for dim in ["Pattern", "Color", "Shape"]:
                if row[f"Item{i}{dim}"] == row[f"Last{dim}"]:
                    num_prev += 1
            if num_prev ==2:
                has_two_idx = i
            if num_prev == 1:
                has_one_idx = i
            row[f"Item{i}PrevFeatures"] = num_prev
            row["HasTwoIdx"] = has_two_idx
            row["HasOneIdx"] = has_one_idx
        return row
    next_trials_prev = next_trials.apply(get_num_prev_features, axis=1).reset_index()
    return next_trials_prev

def get_chosen_two_out_of_has_two(behavioral_data):
    first_corrects = get_first_correct_in_blocks(behavioral_data)
    next_trials_prev = get_trial_after_first_correct_stats(behavioral_data, first_corrects)
    has_two = next_trials_prev[next_trials_prev.HasTwoIdx != -1]
    chosen_two = has_two[has_two.ItemChosen.astype(int) == has_two.HasTwoIdx]
    chosen_one = has_two[has_two.ItemChosen.astype(int) == has_two.HasOneIdx]
    return (len(chosen_one), len(chosen_two), len(has_two))

def get_num_prev_corrects(beh_df):
    responses = beh_df.Response.values
    results = np.empty(len(responses))
    for pos, el in enumerate(responses):
        idx = pos - 1
        num_corrects = 0
        while idx >= 0:
            if responses[idx] == "Correct":
                num_corrects += 1
                idx -= 1
            else:
                break
        results[pos] = num_corrects
    return results

def get_stay_switch_trials(fs, species, subject, exp, session, enforce_inc=True, lfp_trials=True):
    '''
    Use enforce_inc==True to make sure that there is an incorrect trial before the perseveration trial
    Use lfp_trials==True to use only trials used in LFP analysis
    
    switch1 is the first trial after perseveration ends
    switch0 is the trial before switch 1
    switch2 is the trial after switch 1
    '''
    of = startup.get_object_features(fs, species, subject, exp, session)
    
    switch1 = []
    for b in np.unique(of.BlockNumber.values):
        of_sub = of[(of['BlockNumber']==b) & (of['Response'].isin(['Correct', 'Incorrect']))]
        trials = of_sub[of_sub.Perseveration==0].TrialNumber.values

        if len(trials)>0:
            t1 = trials[0]
            if enforce_inc:
                res = of_sub[of_sub['TrialNumber']<t1].Response.values

                if np.any(res=='Incorrect'):
                    switch1.append(t1)
            else:
                switch1.append(t1)
    switch1 = np.array(switch1)
    switch0 = switch1.copy()-1
    switch2 = switch1.copy()+1
    
    if lfp_trials:
        df = startup.get_behavior(fs, species, subject, exp, session, import_obj_features=False)

        df_sub = df[
            (df['response'].isin([200,206])) & \
            (df['act']=='fb') & \
            (df['ignore']==0) & \
            (df['badTrials']==0) & \
            (df['badGroup']==0) & \
            (df['group']>1)
        ]
        
        switch0_cor = df_sub[(df_sub['trial'].isin(switch0)) & (df_sub['response']==200)].trial.values
        switch1_cor = df_sub[(df_sub['trial'].isin(switch1)) & (df_sub['response']==200)].trial.values
        switch2_cor = df_sub[(df_sub['trial'].isin(switch2)) & (df_sub['response']==200)].trial.values
        
        switch0_inc = df_sub[(df_sub['trial'].isin(switch0)) & (df_sub['response']==206)].trial.values
        switch1_inc = df_sub[(df_sub['trial'].isin(switch1)) & (df_sub['response']==206)].trial.values
        switch2_inc = df_sub[(df_sub['trial'].isin(switch2)) & (df_sub['response']==206)].trial.values
    else:
        switch0_cor = of[(of['TrialNumber'].isin(switch0)) & (of['Response']=='Correct')].TrialNumber.values
        switch1_cor = of[(of['TrialNumber'].isin(switch1)) & (of['Response']=='Correct')].TrialNumber.values
        switch2_cor = of[(of['TrialNumber'].isin(switch2)) & (of['Response']=='Correct')].TrialNumber.values
        
        switch0_inc = of[(of['TrialNumber'].isin(switch0)) & (of['Response']=='Incorrect')].TrialNumber.values
        switch1_inc = of[(of['TrialNumber'].isin(switch1)) & (of['Response']=='Incorrect')].TrialNumber.values
        switch2_inc = of[(of['TrialNumber'].isin(switch2)) & (of['Response']=='Incorrect')].TrialNumber.values
    
    return(switch0_cor, switch1_cor, switch2_cor, switch0_inc, switch1_inc, switch2_inc)




