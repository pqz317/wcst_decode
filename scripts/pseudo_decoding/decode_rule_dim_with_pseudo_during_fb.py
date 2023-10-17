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

from trial_splitters.rule_condition_block_splitter import RuleConditionBlockSplitter


PRE_INTERVAL = 1300
POST_INTERVAL = 1500
INTERVAL_SIZE = 100
EVENT = "FeedbackOnset"

rule_to_dim = {
    'CIRCLE': 'SHAPE', 
    'SQUARE': 'SHAPE', 
    'STAR': 'SHAPE', 
    'TRIANGLE': 'SHAPE', 
    'CYAN': 'COLOR', 
    'GREEN': 'COLOR', 
    'MAGENTA': 'COLOR', 
    'YELLOW': 'COLOR', 
    'ESCHER': 'PATTERN', 
    'POLKADOT': 'PATTERN', 
    'RIPPLE': 'PATTERN', 
    'SWIRL': 'PATTERN'
}

feature_dims = ["COLOR", "SHAPE", "PATTERN"]


def create_session_data(sess_name):
    behavior_path = f"/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
    beh = pd.read_csv(behavior_path)
    valid_beh = behavioral_utils.get_valid_trials(beh)
    valid_beh["RuleDim"] = valid_beh.apply(lambda x: rule_to_dim[x.CurrentRule], axis=1)
    print(sess_name)
    # trials = behavioral_utils.get_last_n_corrects_per_block(valid_beh, 5)
    # trials = behavioral_utils.get_not_figured_out_trials(valid_beh)
    # just look at correct trials in the blocks

    trials = valid_beh[valid_beh.Response == "Correct"]
    frs = pd.read_pickle(f"/data/patrick_scratch/multi_sess/{sess_name}/{sess_name}_firing_rates_{PRE_INTERVAL}_{EVENT}_{POST_INTERVAL}_{INTERVAL_SIZE}_bins.pickle")

    # splits with 3 distinct conditions, at least 3 blocks for each rule dimension
    # if unable to generate splits, return None
    try: 
        splitter = RuleConditionBlockSplitter(trials, condition="RuleDim", num_distinct_conditions=3, num_blocks_per_cond=3)
    except ValueError:
        print(f"Unable to create splitter for sess {sess_name}, skipping")
        return None

    return SessionData(sess_name, trials, frs, splitter)

# def check_sess(x):
#     sess_name = x.session_name
#     behavior_path = f"/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
#     beh = pd.read_csv(behavior_path)
#     last_block = beh.BlockNumber.max()
#     valid_beh = beh[
#         (beh.Response.isin(["Correct", "Incorrect"])) & 
#         (beh.BlockNumber >= 2) &
#         (beh.BlockNumber != last_block) 
#     ]  
#     valid_beh["RuleDim"] = valid_beh.apply(lambda x: rule_to_dim[x.CurrentRule], axis=1)

#     grouped_blocks = valid_beh.groupby(by="RuleDim").apply(lambda x: len(x.BlockNumber.unique()) >=2)
#     at_least_two = np.all(grouped_blocks)
#     return at_least_two

def decode_rule_dim(valid_sess):
    sess_datas = valid_sess.apply(lambda x: create_session_data(x.session_name), axis=1)
    print(f"{len(sess_datas)} total sessions considered")
    sess_datas = sess_datas.dropna()
    print(f"{len(sess_datas)} sessions meeting criteria")
    num_neurons = sess_datas.apply(lambda x: x.get_num_neurons()).sum()
    print(f"{num_neurons} neurons to decode with")
    init_params = {"n_inputs": num_neurons, "p_dropout": 0.5, "n_classes": len(feature_dims)}
    trainer = Trainer(learning_rate=0.05, max_iter=1000, batch_size=1000)
    model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, feature_dims)
    time_bins = np.arange(0, 2.8, 0.1)
    train_accs, test_accs, shuffled_accs, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(model, sess_datas, time_bins, 5, 2000, 400, 42)

    base_dir = "/data/patrick_scratch/pseudo"
    np.save(os.path.join(base_dir, f"rule_dim_cor_only_train_accs.npy"), train_accs)
    np.save(os.path.join(base_dir, f"rule_dim_cor_only_test_accs.npy"), test_accs)
    np.save(os.path.join(base_dir, f"rule_dim_cor_only_shuffled_accs.npy"), shuffled_accs)
    np.save(os.path.join(base_dir, f"rule_dim_cor_only_models.npy"), models)


def main():
    valid_sess = pd.read_pickle("/data/patrick_scratch/multi_sess/valid_sessions.pickle")
    decode_rule_dim(valid_sess)

if __name__ == "__main__":
    main()