import numpy as np
import pandas as pd
from lfp_tools import startup
from constants.behavioral_constants import *
from constants.decoding_constants import *
import os


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

def get_valid_trials_sa(beh):
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

def get_valid_trials_blanche(beh):
    last_block = beh.BlockNumber.max()
    valid_beh = beh[
        (beh.Response.isin(["Correct", "Incorrect"])) & 
        (beh.BlockNumber >= 1) &
        (beh.BlockNumber != last_block)
    ]
    return valid_beh

def get_valid_trials(beh, sub="SA"):
    if sub == "SA":
        return get_valid_trials_sa(beh)
    elif sub == "BL":
        return get_valid_trials_blanche(beh)
    else: 
        raise ValueError("unknown subject")
    
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
        renames[f"feat_{i}"] = f"{feat_name}Value"
    model_vals = model_vals.rename(columns=renames)
    valid_beh_vals = pd.merge(beh, model_vals, left_on="TrialNumber", right_on="trial", how="inner")
    # check 
    assert(len(valid_beh_vals) == len(beh))
    def get_highest_val_feat(row):
        color = row["Color"]
        shape = row["Shape"]
        pattern = row["Pattern"]
        vals = {color: row[color + "Value"], shape: row[shape + "Value"], pattern: row[pattern + "Value"]}
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

def balance_trials_by_condition(beh, condition_columns, seed=None, min=None):
    """
    Balance the number of trials for each condition by choosing 
    the minimum number of trials 
    """
    if min is None: 
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
        block["TrialInBlock"] = range(len(block))
        return block
    beh = beh.groupby("BlockNumber", group_keys=False).apply(get_block_lengths).reset_index()
    beh["BlockPosition"] = beh.TrialInBlock / beh.BlockLength
    # beh["BlockPosition"] = beh.TrialAfterRuleChange / beh.BlockLength
    if num_bins:
        beh["BlockPositionBin"] = (beh.BlockPosition * num_bins).astype(int)
    return beh

def get_prev_choice_fbs(beh):
    beh["PrevResponse"] = beh.Response.shift()
    for dim in FEATURE_DIMS:
        beh[f"Prev{dim}"] = beh[dim].shift()
    return beh[~beh.PrevResponse.isna()]

def get_next_choice_fbs(beh):
    beh["NextResponse"] = beh.Response.shift(-1)
    for dim in FEATURE_DIMS:
        beh[f"Next{dim}"] = beh[dim].shift(-1)
    return beh[~beh.NextResponse.isna()]

def get_max_feature_value(beh, num_bins=None, quantize_bins=False):
    """
    Finds the max value'd feature for each trial, assumes beh df already has feature values
    If num_bins specified, adds additional column of value bin of equal size
    If quantize_bins is true, creates value bins with quantiles instead
    """
    def find_max(row):
        max_feat = None
        max_val = 0
        for feat in FEATURES:
            if row[f"{feat}Value"] >= max_val:
                max_val = row[f"{feat}Value"]
                max_feat = feat
        row["MaxFeat"] = max_feat
        row["MaxFeatDim"] = FEATURE_TO_DIM[max_feat]
        row["MaxValue"] = max_val
        return row        
    beh = beh.apply(find_max, axis=1)
    if num_bins:
        if quantize_bins:
            beh["MaxValueBin"] = pd.qcut(beh["MaxValue"], num_bins, labels=False)
        else:
            beh["MaxValueBin"] = pd.cut(beh["MaxValue"], num_bins, labels=False)
    return beh

def calc_feature_probs(beh, beta=1):

    logits = np.empty((len(FEATURES), len(beh)))
    for i, feat in enumerate(FEATURES):
        logits[i, :] = np.exp(beh[f"{feat}Value"] * beta)
    probs = logits / np.sum(logits, axis=0)
    for i, feat in enumerate(FEATURES):
        beh[f"{feat}Prob"] = probs[i, :]
    return beh


def calc_feature_value_entropy(beh, num_bins=None, quantize_bins=False):
    """
    Calculates the feature value entropy for each trial
    Requires beh to have Prob per feature
    """
    sums = np.zeros(len(beh))
    for feat in FEATURES:
        sums += beh[f"{feat}Prob"] * np.log(beh[f"{feat}Prob"])
    beh[f"FeatEntropy"] = -1 * sums
    if num_bins:
        if quantize_bins:
            beh["FeatEntropyBin"] = pd.qcut(beh["FeatEntropy"], num_bins, labels=False)
        else:
            beh["FeatEntropyBin"] = pd.cut(beh["FeatEntropy"], num_bins, labels=False)
    return beh

def calc_confidence(beh, num_bins=None, quantize_bins=False):
    """
    Calcs confidence, just normalized 1/ Entropy. 
    requires beh to have FeatEntropy
    """
    conf = 1 / beh["FeatEntropy"]
    beh["Confidence"] = (conf - conf.min()) / (conf.max() - conf.min())
    if num_bins:
        if quantize_bins:
            beh["ConfidenceBin"] = pd.qcut(beh["Confidence"], num_bins, labels=False)
        else:
            beh["ConfidenceBin"] = pd.cut(beh["Confidence"], num_bins, labels=False)
    return beh


def zscore_feature_vals_by_block(beh, num_bins=None, quantize_bins=False):
    def zscore(block):
        val_cols = [f"{feat}Value" for feat in FEATURES]
        all_vals = pd.melt(block, value_vars=val_cols)
        mean = all_vals.value.mean()
        std = all_vals.value.std()
        # if std is 0, just set to 0. 
        for feat in FEATURES:
            block[f"{feat}ValueBlockZ"] = np.nan_to_num((block[f"{feat}Value"] - mean) / std)
        block["MaxValueBlockZ"] = np.nan_to_num((block["MaxValue"] - mean) / std)
        return block
    beh = beh.groupby("BlockNumber").apply(zscore).reset_index(drop=True)
    if num_bins:
        if quantize_bins:
            beh["MaxValueBlockZBin"] = pd.qcut(beh["MaxValueBlockZ"], num_bins, labels=False)
        else:
            beh["MaxValueBlockZBin"] = pd.cut(beh["MaxValueBlockZ"], num_bins, labels=False)
    return beh

def shuffle_beh_by_shift(beh, column="TrialNumber", buffer=50, rng=None, seed=None):
    if rng is None: 
        rng = np.random.default_rng(seed)
    col_vals = beh[column].values
    shift_idx = rng.integers(buffer, len(beh) - buffer, 1)
    shuffled_col_vals = np.roll(col_vals, shift_idx)
    beh[column] = shuffled_col_vals
    return beh

def shuffle_beh_random(beh, column="TrialNumber", rng=None, seed=None):
    if rng is None: 
        rng = np.random.default_rng(seed)
    col_vals = beh[column].values
    rng.shuffle(col_vals)
    beh[column] = col_vals
    return beh

def shuffle_block_rules(beh, rng=None, seed=None):
    """
    Shuffles current rules by block. 
    trials in the same block still have the same CurrentRule, 
    they're just shuffled across blocks
    """
    if rng is None: 
        rng = np.random.default_rng(seed)
    def get_rule_of_block(block):
        rule = block.CurrentRule.unique()[0]
        return rule
    block_to_rule = beh.groupby("BlockNumber").apply(get_rule_of_block).reset_index(name="CurrentRule")
    labels = block_to_rule.CurrentRule.values.copy()
    rng.shuffle(labels)
    block_to_rule["CurrentRule"] = labels
    beh = beh.drop(columns=["CurrentRule"])  # drop original current rules from beh. 
    beh = pd.merge(beh, block_to_rule, on="BlockNumber")
    # new current rules are shuffled by block
    return beh

def filter_blocks_by_rule_occurence(beh, min_blocks_per_rule):
    """
    Get trials that belong in blocks, where rule occurs in at least min_blocks_per_rule blocks
    """
    num_rule_blocks = beh.groupby("CurrentRule").BlockNumber.nunique()
    rules_meets_condition = num_rule_blocks[num_rule_blocks >= min_blocks_per_rule]
    return beh[beh.CurrentRule.isin(rules_meets_condition.index)]

def filter_max_feat_correct(beh):
    """
    Get trials where the predicted max-valued feature is the rule, and trial was correct
    """
    return beh[
        (beh.CurrentRule == beh.MaxFeat) &
        (beh.Response == "Correct")
    ]

def filter_max_feat_chosen(beh):
    def get_max_feat_chosen(row):
        dim = FEATURE_TO_DIM[row.MaxFeat]
        return row[dim] == row.MaxFeat
    beh["MaxFeatChosen"] = beh.apply(get_max_feat_chosen, axis=1)
    beh = beh[beh.MaxFeatChosen]
    return beh


def get_prob_correct_by_block_pos(beh, max_block_pos):
    """
    get probability of choosing correct as a function of position in block
    """
    def get_block_lengths(block):
        block["BlockLength"] = len(block)
        block["TrialInBlock"] = range(len(block))
        return block
    beh = beh.groupby("BlockNumber", group_keys=False).apply(get_block_lengths).reset_index()
    beh = beh[beh.TrialInBlock < max_block_pos]
    def calc_prob_correct(group):
        return len(group[group.Response == "Correct"]) / len(group)
    prob_correct = beh.groupby("TrialInBlock", group_keys=False).apply(calc_prob_correct).reset_index(name='ProbCorrect')
    return prob_correct

def get_prob_choosing_rule_relative_to_rule_switch(beh, window):
    """
    In some window before/after a rule switch, get probability of choosing a card with that rule
    """
    beh = get_prev_rule(beh)
    beh["ChoseLastRule"] = beh.apply(lambda row: row[FEATURE_TO_DIM[row["PrevRule"]]] == row["PrevRule"] if row["PrevRule"] else False, axis=1)
    beh["ChoseCurrentRule"] = beh.apply(lambda row: row[FEATURE_TO_DIM[row["CurrentRule"]]] == row["CurrentRule"] if row["CurrentRule"] else False, axis=1)

    beh['LastPos'] = beh.index.where(beh.TrialAfterRuleChange == 0)
    beh['LastPos'] = beh['LastPos'].ffill()
    beh['SinceLast'] = beh.index - beh['LastPos']

    beh['NextPos'] = beh.index.where(beh.TrialAfterRuleChange == 0)
    beh['NextPos'] = beh['NextPos'].bfill()
    beh['UntilNext'] = beh['NextPos'] - beh.index

    # filter edges
    beh = beh[(~beh.SinceLast.isna()) & (~beh.UntilNext.isna())]

    since_beh = beh[beh.SinceLast < window]
    chose_last_probs = since_beh.groupby("SinceLast").apply(lambda x: len(x[x.ChoseLastRule]) / len(x)).reset_index(name="ChoiceProb")
    chose_last_probs["Position"] = chose_last_probs["SinceLast"].astype(int)


    until_beh = beh[beh.UntilNext < window]
    chose_next_probs = until_beh.groupby("UntilNext").apply(lambda x: len(x[x.ChoseCurrentRule]) / len(x)).reset_index(name="ChoiceProb")
    chose_next_probs["Position"] = (-1 * chose_next_probs["UntilNext"]).astype(int)

    prob_pos =  pd.concat((chose_last_probs, chose_next_probs))
    return prob_pos


def get_prev_rule(beh):
    if "PrevRule" in beh: 
        return beh
    group_rules =  beh.groupby("BlockNumber", group_keys=True).apply(lambda group: pd.Series({"CurrentRule": group.CurrentRule.iloc[0]})).reset_index()
    group_rules["PrevRule"] = group_rules["CurrentRule"].shift()
    beh = pd.merge(beh, group_rules[["BlockNumber", "PrevRule"]], on="BlockNumber")
    return beh 

def get_perseveration(beh):
    beh["Perseveration"] = beh.apply(lambda row: row[FEATURE_TO_DIM[row["PrevRule"]]] == row["PrevRule"] if row["PrevRule"] else False, axis=1)
    return beh

def get_prob_perseveration_by_block_pos(beh, max_block_pos):
    """
    get probability of choosing correct as a function of position in block
    """
    beh = get_prev_rule(beh)
    beh = get_perseveration(beh)
    def get_block_lengths(block):
        block["BlockLength"] = len(block)
        block["TrialInBlock"] = range(len(block))
        return block
    beh = beh.groupby("BlockNumber", group_keys=False).apply(get_block_lengths).reset_index()
    beh = beh[beh.TrialInBlock < max_block_pos]
    def calc_prob_perseverate(group):
        return len(group[group.Perseveration]) / len(group)
    prob_correct = beh.groupby("TrialInBlock", group_keys=False).apply(calc_prob_perseverate).reset_index(name='Prob Perseverate')
    return prob_correct

def is_choosing_prev_rule(row, prev_rule):
    if prev_rule is None:
        return False
    rule_dim = FEATURE_TO_DIM[prev_rule]
    return row[rule_dim] == prev_rule


def get_perseveration_trials(beh):
    # define perseveration trials as ones where subject 
    # still chooses card that contains previous rule after a block switch
    beh["Perseverating"] = False
    beh["CurrentInferredRule"] = beh["CurrentRule"]
    blocks = beh.BlockNumber.unique()
    prev_rule = None
    for block in blocks: 
        trials = beh[beh.BlockNumber == block]
        for i, row in trials.iterrows():
            if is_choosing_prev_rule(row, prev_rule):
                print("here")
                beh.loc[i, "Perseverating"] = True
                beh.loc[i, "CurrentInferredRule"] = prev_rule
            else: 
                break
        prev_rule = beh.CurrentRule.iloc[0]
    return beh
            
def get_chosen_preferred_trials(pair, beh, high_val_only=True):
    """
    Find trials where either features in the pair are preferred, (high conf for that feature)
    and chosen. 
    NOTE: requries belief state value labels to be included in beh
    """
    feat1, feat2 = pair
    chosen_preferred = beh[
        ((beh[FEATURE_TO_DIM[feat1]] == feat1) & (beh.PreferredBelief == feat1)) |
        ((beh[FEATURE_TO_DIM[feat2]] == feat2) & (beh.PreferredBelief == feat2))
    ]
    if high_val_only:
        chosen_preferred = chosen_preferred[chosen_preferred.BeliefStateValueBin == 1]
    chosen_preferred["Choice"] = chosen_preferred.PreferredBelief
    return chosen_preferred

def get_chosen_not_preferred_trials(pair, beh, high_val_only=True):
    """
    Find trials where either features in the pair are chosen, but are not preferred. 
    Additionally, require that they are high confidence, 
    and the features are not chosen in the same trial. 
    Add additional Choice column
    """
    feat1, feat2 = pair
    # not_pref_beh = beh[
    #     (~beh.BeliefStateValueLabel.isin([f"High {feat1}", f"High {feat2}"])) & 
    #     (beh.BeliefStateValueLabel != "Low")
    # ]

    # find trials where neither features are preferred
    not_pref_beh = beh[~beh.PreferredBelief.isin([feat1, feat2])]
    # whether to filter by high valued trials
    if high_val_only:
        not_pref_beh = not_pref_beh[not_pref_beh.BeliefStateValueBin == 1]

    # chose feature 1 and not 2
    chose_feat_1_not_pref = not_pref_beh[
        (not_pref_beh[FEATURE_TO_DIM[feat1]] == feat1) & 
        (not_pref_beh[FEATURE_TO_DIM[feat2]] != feat2)
    ]
    chose_feat_1_not_pref["Choice"] = feat1
    # vice versa
    chose_feat_2_not_pref = not_pref_beh[
        (not_pref_beh[FEATURE_TO_DIM[feat2]] == feat2) & 
        (not_pref_beh[FEATURE_TO_DIM[feat1]] != feat1)
    ]
    chose_feat_2_not_pref["Choice"] = feat2
    return pd.concat((chose_feat_1_not_pref, chose_feat_2_not_pref))

def get_chosen_single(feat, beh):
    beh["Choice"] = beh.apply(lambda x: feat if x[FEATURE_TO_DIM[feat]] == feat else "other", axis=1)
    beh["FeatPreferred"] = beh["PreferredBelief"].apply(lambda x: "Preferred" if x == feat else "Not Preferred")
    return beh

def get_chosen_preferred_single(feat, beh, high_val_only=False):
    """
    Find trials where either the feature is chosen and preferred, 
    or somet other feature was preferred, and chosen
    """
    # chose feature, high belief in feature
    chose_feat_pref = beh[
        (beh[FEATURE_TO_DIM[feat]] == feat) & 
        (beh.PreferredBelief == feat)
    ]
    chose_feat_pref["Choice"] = feat
    # chose feature, high belief in something else
    chose_other = beh[
        (beh[FEATURE_TO_DIM[feat]] != feat) & 
        (beh.PreferredBelief != feat)
    ]
    chose_other["Choice"] = "other"
    res = pd.concat((chose_feat_pref, chose_other))
    if high_val_only:
        res = res[res.BeliefStateValueBin == 1]
    return res

def get_chosen_not_preferred_single(feat, beh, high_val_only=False):
    """
    Find trials where either the feature is chosen but not preferred, 
    or trials where some other feature was preferred, and chosen
    """
    # trials still have high value, and feature is chosen but not preferred
    chose_feat_not_pref = beh[
        (beh[FEATURE_TO_DIM[feat]] == feat) & 
        (beh.PreferredBelief != feat)
    ]
    chose_feat_not_pref["Choice"] = feat

    # trials have high value, feature is not chosen, also not preferred
    chose_other = beh[
        (beh[FEATURE_TO_DIM[feat]] != feat) & 
        (beh.PreferredBelief != feat)
    ]
    chose_other["Choice"] = "other"
    res = pd.concat((chose_feat_not_pref, chose_other))
    if high_val_only:
        res = res[res.BeliefStateValueBin == 1]
    return res

def get_beliefs_per_session(beh, session_name, subject="SA", base_dir="/data/patrick_res"):
    beliefs_path = os.path.join(base_dir, f"behavior/models/belief_state_agent/sub-{subject}/{session_name}_beliefs.pickle")
    beliefs_df = pd.read_pickle(beliefs_path)
    return pd.merge(beh, beliefs_df, on="TrialNumber")

def shift_beliefs(beh):
    beh[[f"{feat}Prob" for feat in FEATURES]] = beh[[f"{feat}Prob" for feat in FEATURES]].shift(-1)
    beh["BeliefStateValue"] = beh.BeliefStateValue.shift(-1)
    beh = beh[~beh.BeliefStateValue.isna()]
    return beh

def get_belief_value_labels(beh):
    med = beh.BeliefStateValue.median()
    beh["BeliefStateValueBin"] = beh.apply(lambda x: 0 if x.BeliefStateValue < med else 1, axis=1)
    beh["PreferredBelief"] = beh[[f"{feat}Prob" for feat in FEATURES]].idxmax(axis=1).apply(lambda x: x[:-4])
    beh["PreferredBeliefProb"] = beh.apply(lambda x: x[f"{x.PreferredBelief}Prob"], axis=1)
    beh["BeliefStateValueLabel"] = beh.apply(lambda x: f"High {x.PreferredBelief}" if x.BeliefStateValueBin == 1 else "Low", axis=1)
    beh["PreferredChosen"] = beh.apply(lambda x: x[FEATURE_TO_DIM[x.PreferredBelief]] == x.PreferredBelief, axis=1)
    return beh

def get_belief_partitions(beh, feat, use_x=False, thresh=BELIEF_PARTITION_THRESH):
    """
    Adds two additional columns to be df
    Partitions belief state space into: 
     - Low (all beliefs < thresh)
     - High X where X is the feature of interest, when X has the highest belief
     - High Not X
    Per trial (row), adds
     - BeliefConf (Low or High)
     - BeliefPartition (Low, High X, High Not X)
    Assumes df already contains columns: 
    - PreferredBeliefProb, PreferredBelief
    """
    def label_trial(row):
        feat_str = "X" if use_x else feat
        if row.PreferredBeliefProb <= thresh: 
            return pd.Series(["Low", f"Not {feat_str}", "Low"])
        elif row.PreferredBelief == feat: 
            return pd.Series(["High", feat_str, f"High {feat_str}"])
        else: 
            return pd.Series(["High", f"Not {feat_str}", f"High Not {feat_str}"])
    beh[["BeliefConf", "BeliefPolicy", "BeliefPartition"]] = beh.apply(label_trial, axis=1)
    beh[["NextBeliefConf", "NextBeliefPolicy", "NextBeliefPartition"]] = beh[["BeliefConf", "BeliefPolicy", "BeliefPartition"]].shift(-1)
    beh = beh[~beh.NextBeliefConf.isna()]
    return beh
    

def get_good_pairs_across_sessions(all_beh, block_thresh):
    """
    Find pairs of features, and associated sessions where
    each item in the pair shows up in the session at least 
    block_thresh number of blocks. 
    Input: 
        all_beh, concatenated beh with sessions column
    output: 
        df of pair, sessions, num_sessions, dim_type
    """
    num_blocks = all_beh.groupby(["session", "CurrentRule"]).apply(lambda x: len(x.BlockNumber.unique())).reset_index()
    pairs = []
    for i in range(12):
        for j in range(i + 1, 12):
            feat1 = FEATURES[i]
            feat2 = FEATURES[j]
            sess_1 = num_blocks[(num_blocks.CurrentRule == feat1) & (num_blocks[0] >= block_thresh)].session
            sess_2 = num_blocks[(num_blocks.CurrentRule == feat2) & (num_blocks[0] >= block_thresh)].session
            joints = sess_1[sess_1.isin(sess_2)].values
            if FEATURE_TO_DIM[feat1] == FEATURE_TO_DIM[feat2]:
                dim_type = "within dim"
            else: 
                dim_type = "across dim"
            pairs.append({"pair": [feat1, feat2], "sessions": joints, "num_sessions": len(joints), "dim_type": dim_type})
    pairs = pd.DataFrame(pairs)
    return pairs

def get_good_sessions_per_rule(all_beh, block_thresh):
    good_sess = []
    num_blocks = all_beh.groupby(["session", "CurrentRule"]).apply(lambda x: len(x.BlockNumber.unique())).reset_index()
    for feat in FEATURES:
        sessions = num_blocks[(num_blocks.CurrentRule == feat) & (num_blocks[0] >= block_thresh)].session.values
        good_sess.append({"feat": feat, "sessions": sessions, "num_sessions": len(sessions)})
    return pd.DataFrame(good_sess)
    

def get_valid_belief_beh_for_sub_sess(sub, session):
    behavior_path = SESS_BEHAVIOR_PATH.format(
        sess_name=session,
        sub=sub
    )
    beh = pd.read_csv(behavior_path)
    beh = get_valid_trials(beh, sub)
    feature_selections = get_selection_features(beh)
    beh = pd.merge(beh, feature_selections, on="TrialNumber", how="inner")
    beh = get_beliefs_per_session(beh, session, sub)
    beh = get_belief_value_labels(beh)
    beh = get_prev_choice_fbs(beh)
    beh["session"] = session
    return beh

def filter_behavior(beh, filters):
    """
    filters behavior based on dict of filters, keys of colums values of values
    """
    for col, val in filters.items():
        beh = beh[beh[col] == val]
    return beh

def get_sub_for_session(session):
    """
    Hacky way to get the subject from only a session
    Works because all fo BL sessions recorded during 2019
    """
    # only want first 8 characters due to sessions like 201807250001
    session = int(str(session)[:8])
    return "SA" if session < 20190101 else "BL"


def shuffle_beh_by_session_permute(beh, session, args):
    """
    Shuffles behavior by permuting sessions
    """
    other_sessions = [s for s in args.all_sessions.session_name if s != session]
    print(f"Session permutation shuffle set, randomly choosing from {len(other_sessions)} other sessions")
    
    seed = int(session) * 100 + args.shuffle_idx
    rng = np.random.default_rng(seed)
    other_session = rng.choice(other_sessions)
    other_sub = get_sub_for_session(other_session)
    other_beh = get_valid_belief_beh_for_sub_sess(other_sub, other_session)    
    min_trials = np.min((len(beh), len(other_beh)))
    other_beh = other_beh[:min_trials]
    other_beh["TrialNumber"] = sorted(beh[:min_trials].TrialNumber.unique())
    return other_beh

def load_behavior_from_args(session, args):
    """
    Utility to load behavior, given some Namespace of arguments
    If shuffle flags are set, also does the shuffling
    NOTE: This function does not do filtering by conditions, other than
    to get rid of invalid trials. 
    args: 
      - shuffle_method: either session_permute, circular_shift, random
      - sessions: needs to be specified if session_permute shuffle method
      - shuffle_idx: and int or None
    session: session name to load
    """
    beh = get_valid_belief_beh_for_sub_sess(args.subject, session)
    if args.shuffle_idx is not None:
        if args.shuffle_method == "circular_shift":
            beh = shuffle_beh_by_shift(beh, buffer=50, seed=args.shuffle_idx)
        elif args.shuffle_method == "random":
            beh = shuffle_beh_random(beh, seed=args.shuffle_idx)
        elif args.shuffle_method == "session_permute":
            beh = shuffle_beh_by_session_permute(beh, session, args)
        else:
            raise ValueError(f"shuffle idx is set: {args.shuffle_idx} but invalid shuffle method: {args.shuffle_method}")
    return beh

def get_feat_choice_label(beh, feat):
    beh["Choice"] = beh.apply(lambda x: "Chose" if x[FEATURE_TO_DIM[feat]] == feat else "Not Chose", axis=1)
    return beh

def get_prev_feat_choice_label(beh, feat):
    beh["PrevChoice"] = beh.apply(lambda x: "Chose" if x[f"Prev{FEATURE_TO_DIM[feat]}"] == feat else "Not Chose", axis=1)
    return beh

def get_next_feat_choice_label(beh, feat):
    beh["NextChoice"] = beh.apply(lambda x: "Chose" if x[f"Next{FEATURE_TO_DIM[feat]}"] == feat else "Not Chose", axis=1)
    return beh

def get_label_by_mode(beh, mode):
    """
    Adds a column called label to beh df, populates it depending on mode. 
    Potentially filters beh df as well. 
    """
    if mode == "conf":
        beh["condition"] = beh["BeliefConf"]
    elif mode == "policy":
        beh["condition"] = beh["BeliefPolicy"]
    elif mode == "pref":
        beh = beh[beh.BeliefPartition.isin(["High X", "High Not X"])]
        beh["condition"] = beh["BeliefPartition"]
    elif mode == "feat_belief":
        beh = beh[beh.BeliefPartition.isin(["Low", "High X"])]
        beh["condition"] = beh["BeliefPartition"]
    elif mode == "choice":
        beh["condition"] = beh["Choice"]
    elif mode == "reward":
        beh["condition"] = beh["Response"]
    elif mode in ["chose_and_correct", "reward_int", "choice_int", "updates_beliefs"]:
        beh["condition"] = beh["Choice"] + " " + beh["Response"]
    else: 
        raise ValueError("invalid mode in args")
    return beh

def load_all_beh_for_sub(sub):
    valid_sess = pd.read_pickle(SESSIONS_PATH.format(sub=sub))
    return pd.concat(valid_sess.apply(lambda x: get_valid_belief_beh_for_sub_sess(sub, x.session_name), axis=1).values)