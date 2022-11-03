import numpy as np
import pandas as pd

direction_to_item = {
    "up": 0,
    "down": 1,
    "right": 2,
    "left": 3
}

response_mapping = {
    1: "Correct",
    0: "Incorrect",
}

feat_to_pos = {
    "Shape": 0,
    "Color": 1,
    "Pattern": 2,
}

def add_block_number(beh):
    block_num = 0
    block_nums = []
    for _, row in beh.iterrows():
        if row.rule_shift == "yes":
            block_num += 1
        block_nums.append(block_num)
    beh["BlockNumber"] = block_nums
    return beh

def replace_feature_texture_S(beh):
    for idx, row in beh.iterrows():
        for card_pos in range(1, 5):
            column_name = f"bmp_table_{card_pos}"
            card_features = row[column_name]
            if card_features[2] == "S":
                chars = list(card_features)
                chars[2] = "Z"
                beh.at[idx, column_name] = "".join(chars)
    return beh

def replace_rule_feature_S(beh, subject, session):
    # TODO: fill out with actual implementation
    if subject == "sub-IR84" and session == "sess-1":
        return beh
    else: 
        raise NotImplementedError


def initialize_item_cols(df):
    for i in range(4):
        for feat in ["Color", "Shape", "Pattern"]:
            df[f"Item{i}{feat}"] = []
    return df

def add_item_cols(df, row):
    for i in range(4):
        for feat in ["Color", "Shape", "Pattern"]:
            # column names are indexed by 1
            card = row[f"bmp_table_{i + 1}"]
            card[feat_to_pos[feat]]
            df[f"Item{i}{feat}"].append(card[feat_to_pos[feat]])
    return df

def rename_columns(beh, timestamps):
    # key_resp_2_keys -> ItemChosen
    # bmp_table_* -> Item<Number><Dim>
    # answer_correctness -> Response (Correct/Incorrect)
    # idx -> TrialNumber
    # timestamps.img_offset -> FeedbackOnset
    df = {
        "TrialNumber": [],
        "ItemChosen": [],
        "Response": [],
        "FeedbackOnset": [],
        "CurrentRule": [],
    }
    df = initialize_item_cols(df)
    for _, row in beh.iterrows():
        trial_number = row["trials_thisTrialN"]
        df["TrialNumber"].append(trial_number)
        df["ItemChosen"].append(direction_to_item[row["key_resp_2_keys"]])
        df["Response"].append(response_mapping[row["ans_correctness"]])
        # timestamp trial numbers are indexed by 1
        img_offset = timestamps[timestamps.trial_number == trial_number + 1].img_offset.values[0]
        df["FeedbackOnset"].append(img_offset)
        df["CurrentRule"].append(row["rule"])
        df = add_item_cols(df, row)
    return pd.DataFrame(df)


def fetch_formatted_human_behavior(fs, subject, session):
    beh_path = f"human-lfp/wcst-preprocessed/rawdata/{subject}/{session}/behavior/{subject}-{session}-beh.csv"
    beh = pd.read_csv(fs.open(beh_path)) 

    timestamps_path = f"human-lfp/wcst-preprocessed/rawdata/{subject}/{session}/behavior/{subject}-{session}-trial_timestamps.csv"
    timestamps = pd.read_csv(fs.open(timestamps_path))
    # filter out trials with no choice
    beh = beh[beh.key_resp_2_keys != 'None']
    beh = add_block_number(beh)
    beh = replace_feature_texture_S(beh)
    beh = replace_rule_feature_S(beh, subject, session)
    return rename_columns(beh, timestamps)