# An updated version of decode_features_of_rule_dim.py...
# for each feature dimension filter for blocks where rule matches feature dim
# look at only the last N (5 or 8) correct trials in blocks
# see if "representation" (weights) for feature dimension when the dimension is the rule is
# different from when the dimension is not the rule
# splitters: 
#  - per session, for rule dim: split blocks using rule dim conditions
#  - grab other blocks, just rule a random splitter

# look at decoding features from the only the last 8 corrects in each session, 
# split by blocks and by random trials 
# this is to ensure feature decoding is still possible, 
# and establish a baseline with both splitting techniques

import os
import numpy as np
import pandas as pd
import utils.pseudo_utils as pseudo_utils
import utils.pseudo_classifier_utils as pseudo_classifier_utils
import utils.behavioral_utils as behavioral_utils

from utils.session_data import SessionData

from models.trainer import Trainer
from models.model_wrapper import ModelWrapper, ModelWrapperLinearRegression
from models.multinomial_logistic_regressor import NormedDropoutMultinomialLogisticRegressor

from trial_splitters.condition_trial_splitter import ConditionTrialSplitter 
from trial_splitters.rule_condition_block_splitter import RuleConditionBlockSplitter

PRE_INTERVAL = 1300
POST_INTERVAL = 1500
INTERVAL_SIZE = 100
EVENT = "FeedbackOnset"

feature_dims = ["Color", "Shape", "Pattern"]
possible_features = {
    "Color": ['CYAN', 'GREEN', 'MAGENTA', 'YELLOW'],
    "Shape": ['CIRCLE', 'SQUARE', 'STAR', 'TRIANGLE'],
    "Pattern": ['ESCHER', 'POLKADOT', 'RIPPLE', 'SWIRL']
}
rule_to_dim = {
    'CIRCLE': 'Shape', 
    'SQUARE': 'Shape', 
    'STAR': 'Shape', 
    'TRIANGLE': 'Shape', 
    'CYAN': 'Color', 
    'GREEN': 'Color', 
    'MAGENTA': 'Color', 
    'YELLOW': 'Color', 
    'ESCHER': 'Pattern', 
    'POLKADOT': 'Pattern', 
    'RIPPLE': 'Pattern', 
    'SWIRL': 'Pattern'
}

SPLIT_SEED = 42
MIN_NUM_BLOCKS_PER_RULE = 2
LAST_N_CORRECTS_PER_BLOCK = 8

def create_session_datas(sess_name, feature_dim):
    """
    Per session, will create 2 SessionData objects
        1 with a rule dim splitter, with blocks that match dim
        1 with random splitter, with blocks that don't match dim. 
    """
    behavior_path = f"/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
    beh = pd.read_csv(behavior_path)

    valid_beh = behavioral_utils.get_valid_trials(beh)
    # last_eights = behavioral_utils.get_last_n_corrects_per_block(valid_beh, LAST_N_CORRECTS_PER_BLOCK)
    last_eights = valid_beh[valid_beh.Response == "Correct"]
    feature_selections = behavioral_utils.get_selection_features(last_eights)
    last_eights_merged = pd.merge(last_eights, feature_selections, on="TrialNumber", how="inner")
    last_eights_merged["RuleDim"] = last_eights_merged.apply(lambda x: rule_to_dim[x.CurrentRule], axis=1)
    last_eights_only_rule = last_eights_merged[last_eights_merged.RuleDim == feature_dim]
    last_eights_not_rule = last_eights_merged[last_eights_merged.RuleDim != feature_dim]

    print(sess_name)
    frs = pd.read_pickle(f"/data/patrick_scratch/multi_sess/{sess_name}/{sess_name}_firing_rates_{PRE_INTERVAL}_{EVENT}_{POST_INTERVAL}_{INTERVAL_SIZE}_bins.pickle")
    rule_dim_splitter = RuleConditionBlockSplitter(
        last_eights_only_rule, 
        condition="CurrentRule", 
        seed=SPLIT_SEED,
        num_distinct_conditions=4,
        num_blocks_per_cond=MIN_NUM_BLOCKS_PER_RULE,
    )
    match_dim_sess_data = SessionData(sess_name, last_eights_only_rule, frs, rule_dim_splitter)

    other_splitter = ConditionTrialSplitter(last_eights_not_rule, feature_dim, 0.2)
    not_match_dim_sess_data = SessionData(sess_name, last_eights_not_rule, frs, other_splitter)

    return (match_dim_sess_data, not_match_dim_sess_data)

def decode_features_of_rule_dim(feature_dim, valid_sess):
    print(f"Decoding features where rule is {feature_dim}")
    base_dir = "/data/patrick_scratch/pseudo"
    valid_sess = valid_sess[valid_sess[feature_dim]]
    print(f"{len(valid_sess)} sessions satisfy condition")
    if len(valid_sess) == 0: 
        print("No sessions satisfy condition, returning")
        return
    sess_datas = valid_sess.apply(lambda x: create_session_datas(x.session_name, feature_dim), axis=1)
    # sess_datas = sess_datas.dropna()

    # store sessions that were used
    sess_names = sess_datas.apply(lambda x: x[0].sess_name)  # choice of 0 arbitrary
    sess_names.to_pickle(os.path.join(base_dir, f"rule_dim_{feature_dim}_sessions.pickle"))

    print(f"{len(sess_datas)} sessions successfully generated splits, using them")
    print(f"Feature dim is {feature_dim}")
    names = ["match_rule_dim", "not_match_rule_dim"]
    for i in range(2):
        sess_data = sess_datas.apply(lambda x: x[i])
        classes = possible_features[feature_dim]
        num_neurons = sess_data.apply(lambda x: x.get_num_neurons()).sum()
        init_params = {"n_inputs": num_neurons, "p_dropout": 0.5, "n_classes": len(classes)}
        trainer = Trainer(learning_rate=0.05, max_iter=500, batch_size=1000)
        model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, classes)
        time_bins = np.arange(0, 2.8, 0.1)
        train_accs, test_accs, shuffled_accs, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(model, sess_data, time_bins, 5, 2000, 400, 42)
        # np.save(os.path.join(base_dir, f"{feature_dim}_last_eight_{names[i]}_train_accs.npy"), train_accs)
        # np.save(os.path.join(base_dir, f"{feature_dim}_last_eight_{names[i]}_test_accs.npy"), test_accs)
        # np.save(os.path.join(base_dir, f"{feature_dim}_last_eight_{names[i]}_shuffled_accs.npy"), shuffled_accs)
        # np.save(os.path.join(base_dir, f"{feature_dim}_last_eight_{names[i]}_models.npy"), models)
        np.save(os.path.join(base_dir, f"{feature_dim}_cors_{names[i]}_train_accs_2.npy"), train_accs)
        np.save(os.path.join(base_dir, f"{feature_dim}_cors_{names[i]}_test_accs_2.npy"), test_accs)
        np.save(os.path.join(base_dir, f"{feature_dim}_cors_{names[i]}_shuffled_accs_2.npy"), shuffled_accs)
        np.save(os.path.join(base_dir, f"{feature_dim}_cors_{names[i]}_models_2.npy"), models)       


def main():
    valid_sess = pd.read_pickle("/data/patrick_scratch/multi_sess/valid_sessions_enough_rules_2.pickle")

    for feature_dim in feature_dims: 
        decode_features_of_rule_dim(feature_dim, valid_sess)

if __name__ == "__main__":
    main()