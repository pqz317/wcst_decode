import numpy as np
import pandas as pd
from lfp_tools import startup
from constants.behavioral_constants import *

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

def get_last_five_corrects_per_block(beh):
    """ 
    Filter for trials with:
    - Response is correct
    - Previous response was also correct
    - Per block, grab the last 5 satifying this 
    - Only returns blocks corresponding to rules that occured in at least 3 blocks
    """
    beh["PrevResponse"] = beh.Response.shift()
    cor_cors = beh[(beh.Response == "Correct") & (beh.PrevResponse == "Correct")]
    num_blocks = cor_cors.groupby(by="CurrentRule").apply(lambda g: len(g.BlockNumber.unique()))
    # valid_rules = num_blocks[num_blocks > 3].index
    cor_cor_valid_rules = cor_cors[cor_cors.CurrentRule.isin(num_blocks.index)]
    def last_five(block):
        # last 5 items that meet this critera
        return block.iloc[-5:]
    last_fives = cor_cor_valid_rules.groupby(by="BlockNumber", as_index=False).apply(last_five).reset_index()
    return last_fives

def get_last_n_corrects_per_block(beh, n):
    """ 
    Filter for trials with:
    - Response is correct
    - Per block, grab the last 5 satifying this 
    - Only returns blocks corresponding to rules that occured in at least 3 blocks
    """
    cors = beh[beh.Response == "Correct"]
    def last_n_per_block(block, n):
        # last n items that meet this critera
        return block.iloc[-n:]
    last_ns = cors.groupby(by="BlockNumber", as_index=False).apply(
        lambda x: last_n_per_block(x, n)
    ).reset_index()
    return last_ns

def get_first_n_corrects_per_block(beh, n):
    """ 
    Filter for trials with:
    - Response is correct
    - Per block, grab the first 5 satifying this 
    - Only returns blocks corresponding to rules that occured in at least 3 blocks
    """
    cors = beh[beh.Response == "Correct"]
    def first_n_per_block(block, n):
        # last n items that meet this critera
        return block.iloc[:n]
    last_ns = cors.groupby(by="BlockNumber", as_index=False).apply(
        lambda x: first_n_per_block(x, n)
    ).reset_index()
    return last_ns

def get_valid_trials(beh):
    """
    Filters trials where *usually* are not wanted for decoding, specifically filters out: 
    - any trials that don't result in correct/incorrect response (incomplete or error trials)
    - any trials from the first 2 blocks (these are usually fixed)
    - any trials from the last blocks
    - any blocks with less than 8 trials
    """
    last_block = beh.BlockNumber.max()
    longer_thans = beh.groupby("BlockNumber").apply(lambda x: len(x.TrialNumber.unique()) >= 8)
    longer_than_block_idxs = longer_thans[longer_thans].index
    valid_beh = beh[
        (beh.Response.isin(["Correct", "Incorrect"])) & 
        (beh.BlockNumber >= 2) &
        (beh.BlockNumber != last_block) &
        (beh.BlockNumber.isin(longer_than_block_idxs))
    ]  
    return valid_beh

def get_rpes_per_session(session, beh):
    probs_path = f"/data/082023_RL_Probs/sess-{session}_hv.csv"
    probs = pd.read_csv(probs_path)
    probs["RPE_FE"] = probs.fb - probs.Prob_FE
    probs["RPE_FD"] = probs.fb - probs.Prob_FD
    probs["RPE_FRL"] = probs.fb - probs.Prob_FRL
    merged = pd.merge(beh, probs, left_on="TrialNumber", right_on="trial", how="inner")
    return merged

def get_rpe_groups_per_session(session, beh):
    valid_beh_rpes = get_rpes_per_session(session, beh)
    assert len(valid_beh_rpes) == len(beh)
    pos_med = valid_beh_rpes[valid_beh_rpes.RPE_FE > 0].RPE_FE.median()
    neg_med = valid_beh_rpes[valid_beh_rpes.RPE_FE <= 0].RPE_FE.median()
    # add median labels to 
    def add_group(row):
        rpe = row.RPE_FE
        group = None
        if rpe < neg_med:
            group = "more neg"
        elif rpe >= neg_med and rpe < 0:
            group = "less neg"
        elif rpe >= 0 and rpe < pos_med:
            group = "less pos"
        elif rpe >= pos_med:
            group = "more pos"
        row["RPEGroup"] = group
        return row
    valid_beh_rpes = valid_beh_rpes.apply(add_group, axis=1)
    return valid_beh_rpes

def get_feature_values_per_session(session, beh):
    # need beh to include selected features already
    beh_model_path = f"/data/082023_Feat_RLDE_HV/sess-{session}_hv.csv"
    model_vals = pd.read_csv(beh_model_path)
    renames = {}
    for i, feat_name in enumerate(FEATURES):
        renames[f"feat_{i}"] = feat_name
    model_vals = model_vals.rename(columns=renames)
    valid_beh_vals = pd.merge(beh, model_vals, left_on="TrialNumber", right_on="trial", how="inner")
    # check 
    assert(len(valid_beh_vals) == len(beh))
    def get_highest_val_feat(row):
        color = row["Color"]
        shape = row["Shape"]
        pattern = row["Pattern"]
        vals = {color: row[color], shape: row[shape], pattern: row[pattern]}
        max_feat = max(zip(vals.values(), vals.keys()))[1]
        row["MaxFeat"] = max_feat
        return row
    valid_beh_max = valid_beh_vals.apply(get_highest_val_feat, axis=1)
    return valid_beh_max


def get_min_num_trials_by_condition(beh, condition_columns):
    """
    Get the minimum number of trials per condition
    """
    counts = beh.groupby(condition_columns).count()
    return np.min(counts.TrialNumber)

def validate_enough_trials_by_condition(beh, condition_columns, min_trials, num_unique_conditions=None):
    """
    Check that in behavioral df, groups grouped by condition columns each 
    have more than the min number of trials
    Returns True if condition is satisfied, False if not.
    """
    num_unique_conditions_beh = len(beh.groupby(condition_columns))
    if num_unique_conditions and num_unique_conditions_beh != num_unique_conditions:
        return False
    min = get_min_num_trials_by_condition(beh, condition_columns)
    return min >= min_trials

def balance_trials_by_condition(beh, condition_columns, seed=None):
    """
    Balance the number of trials for each condition by choosing 
    the minimum number of trials 
    """
    min = get_min_num_trials_by_condition(beh, condition_columns)
    sampled = beh.groupby(condition_columns).sample(n=min, random_state=seed)
    return sampled

def get_beh_model_labels_for_session_feat(session, feat, beh_path=SESS_BEHAVIOR_PATH):
    """
    Helper method to add RPE group and Max Feature Matches columns to behavioral df. 
    Adds RPE group and 
    """
    behavior_path = beh_path.format(sess_name=session)
    beh = pd.read_csv(behavior_path)
    valid_beh = get_valid_trials(beh)
    feature_selections = get_selection_features(valid_beh)
    valid_beh_merged = pd.merge(valid_beh, feature_selections, on="TrialNumber", how="inner")
    feat_dim = FEATURE_TO_DIM[feat]
    valid_beh_merged = valid_beh_merged[valid_beh_merged[feat_dim] == feat]
    valid_beh_vals = get_feature_values_per_session(session, valid_beh_merged)
    valid_beh_vals_conf = get_rpe_groups_per_session(session, valid_beh_vals)

    valid_beh_vals_conf["MaxFeatMatches"] = valid_beh_vals_conf.MaxFeat == feat
    valid_beh_vals_conf["Session"] = session
    return valid_beh_vals_conf

def get_relative_block_position(beh, num_bins=None):
    """
    Assigns a relative block position to each trial
    If num_bins specified, also asigns a block position bin
    """
    def get_block_lengths(block):
        block["BlockLength"] = len(block)
        return block
    beh = beh.groupby("BlockNumber").apply(get_block_lengths).reset_index()
    beh["BlockPosition"] = beh.TrialAfterRuleChange / (beh.BlockLength - 1)
    if num_bins:
        beh["BlockPositionBin"] = (beh.BlockPosition * num_bins).astype(int)
    return beh

