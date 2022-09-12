

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

def rename_columns(beh):
    # key_resp_2_keys -> ItemChosen
    # bmp_table_* -> Item<Number><Dim>
    # answer_correctness -> Response (Correct/Incorrect)
    # idx -> TrialNumber
    pass