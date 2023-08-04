# for each feature dimension filter for blocks where rule matches feature dim
# see if feature decoding of rule dimension is better than feature decoding of other dimensions
# splitters: 
#  - per session, for rule dim: split blocks using rule dim conditions
#  - generate n block splits 
#  - store them somewhere
#  - for the other dims: grab the same block splits, change the condition, 

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

from trial_splitters.condition_kfold_block_splitter import ConditionKFoldBlockSplitter
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

split_seed = 42

def create_session_datas(sess_name, rule_dim):
    """
    Per session, will create a tuple (3) of SessionData objects
        1 with a rule dim splitter
        2 with the same blocks, except for the other 2 features
    """
    behavior_path = f"/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
    beh = pd.read_csv(behavior_path)

    valid_beh = behavioral_utils.get_valid_trials(beh)
    last_eights = behavioral_utils.get_last_n_corrects_per_block(valid_beh, 8)
    feature_selections = behavioral_utils.get_selection_features(last_eights)
    last_eights_merged = pd.merge(last_eights, feature_selections, on="TrialNumber", how="inner")
    last_eights_merged["RuleDim"] = last_eights_merged.apply(lambda x: rule_to_dim[x.CurrentRule], axis=1)
    last_eights_only_rule = last_eights_merged[last_eights_merged.RuleDim == rule_dim]

    print(sess_name)
    frs = pd.read_pickle(f"/data/patrick_scratch/multi_sess/{sess_name}/{sess_name}_firing_rates_{PRE_INTERVAL}_{EVENT}_{POST_INTERVAL}_{INTERVAL_SIZE}_bins.pickle")

    # first create a rule condition splitter
    try: 
        rule_dim_splitter = RuleConditionBlockSplitter(
            last_eights_only_rule, 
            condition="CurrentRule", 
            seed=split_seed,
            num_distinct_conditions=4,
        )
    except ValueError as e:
        print(f"Error while generating splits for session {sess_name}, skipping:")
        print(e)
        return None
    rule_dim_sess_data = SessionData(sess_name, last_eights_only_rule, frs, rule_dim_splitter)
    splits = rule_dim_sess_data.pre_generate_splits(5)
    block_splits = [
        (np.concatenate(split.TrainBlocks.values), np.concatenate(split.TestBlocks.values))
        for split in splits
    ]
    # Next create session data for 2 other features
    other_dims = [dim for dim in feature_dims if dim != rule_dim]
    sess_datas_dict = {}
    for dim in other_dims:
        print(f"Generating sess data for dim {dim}")
        try: 
            # try generating splits
            other_block_splitter = ConditionKFoldBlockSplitter(
                last_eights_only_rule, dim, 5, block_splits, 42, 
                min_trials_per_cond=3, num_distinct_conditions=4
            )
            other_sess_data = SessionData(sess_name, last_eights_only_rule, frs, other_block_splitter)
            other_sess_data.pre_generate_splits(5)
            sess_datas_dict[dim] = other_sess_data
        except ValueError as e:
            print(f"Error while generating splits for session {sess_name}, skipping:")
            print(e)
            return None
    sess_datas_dict[rule_dim] = rule_dim_sess_data
    return sess_datas_dict

def decode_features_of_rule_dim(rule_dim, valid_sess):
    print(f"Decoding features where rule is {rule_dim}")
    sess_datas = valid_sess.apply(lambda x: create_session_datas(x.session_name, rule_dim), axis=1)
    sess_datas = sess_datas.dropna()
    print(f"{len(sess_datas)} sessions successfully generated splits, using them")
    for feature_dim in feature_dims:
        print(f"Rule dim is {rule_dim}, decoding {feature_dim}")
        feature_sess_datas = sess_datas.apply(lambda x: x[feature_dim])
        classes = possible_features[feature_dim]
        num_neurons = feature_sess_datas.apply(lambda x: x.get_num_neurons()).sum()
        init_params = {"n_inputs": num_neurons, "p_dropout": 0.5, "n_classes": len(classes)}
        trainer = Trainer(learning_rate=0.05, max_iter=500, batch_size=1000)
        model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, classes)
        time_bins = np.arange(0, 2.8, 0.1)
        train_accs, test_accs, shuffled_accs, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(model, feature_sess_datas, time_bins, 5, 2000, 400, 42)

        base_dir = "/data/patrick_scratch/pseudo"
        np.save(os.path.join(base_dir, f"rule_dim_{rule_dim}_{feature_dim}_decoding_train_accs.npy"), train_accs)
        np.save(os.path.join(base_dir, f"rule_dim_{rule_dim}_{feature_dim}_decoding_test_accs.npy"), test_accs)
        np.save(os.path.join(base_dir, f"rule_dim_{rule_dim}_{feature_dim}_decoding_shuffled_accs.npy"), shuffled_accs)
        np.save(os.path.join(base_dir, f"rule_dim_{rule_dim}_{feature_dim}_decoding_models.npy"), models)


def main():
    valid_sess = pd.read_pickle("/data/patrick_scratch/multi_sess/valid_sessions.pickle")

    for rule_dim in feature_dims: 
        decode_features_of_rule_dim(rule_dim, valid_sess)

if __name__ == "__main__":
    main()