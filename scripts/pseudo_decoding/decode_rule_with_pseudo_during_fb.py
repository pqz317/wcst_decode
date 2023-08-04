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

possible_rules = [
    'CIRCLE', 'SQUARE', 'STAR', 'TRIANGLE', 
    'CYAN', 'GREEN', 'MAGENTA', 'YELLOW', 
    'ESCHER', 'POLKADOT', 'RIPPLE', 'SWIRL'
]

def create_session_data(sess_name):
    behavior_path = f"/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
    beh = pd.read_csv(behavior_path)
    valid_beh = beh[(beh.Response.isin(["Correct", "Incorrect"])) & (beh.BlockNumber >= 2)]  
    print(sess_name)
    cors = valid_beh[valid_beh.Response == "Correct"]
    # print(cors.groupby("CurrentRule").apply(lambda x: len(x.BlockNumber.unique()))) 
    last_fives = behavioral_utils.get_last_n_corrects_per_block(valid_beh, 5)
    # print(last_fives.groupby("CurrentRule").apply(lambda x: len(x.BlockNumber.unique()))) 
    print(last_fives.groupby("CurrentRule").apply(lambda x: len(x)))
    frs = pd.read_pickle(f"/data/patrick_scratch/multi_sess/{sess_name}/{sess_name}_firing_rates_{PRE_INTERVAL}_{EVENT}_{POST_INTERVAL}_{INTERVAL_SIZE}_bins.pickle")
    splitter = RuleConditionBlockSplitter(last_fives)
    return SessionData(sess_name, last_fives, frs, splitter)

def check_sess(x):
    sess_name = x.session_name
    behavior_path = f"/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
    beh = pd.read_csv(behavior_path)
    valid_beh = beh[(beh.Response.isin(["Correct", "Incorrect"])) & (beh.BlockNumber >= 2)]  
    grouped_blocks = valid_beh.groupby(by="CurrentRule").apply(lambda x: len(x.BlockNumber.unique()) >=2)
    has_all_rules = len(grouped_blocks) == 12
    at_least_two = np.all(grouped_blocks)
    return has_all_rules and at_least_two

def decode_rule(valid_sess):
    sess_datas = valid_sess.apply(lambda x: create_session_data(x.session_name), axis=1)

    num_neurons = sess_datas.apply(lambda x: x.get_num_neurons()).sum()
    print(f"{num_neurons} neurons to decode with")
    init_params = {"n_inputs": num_neurons, "p_dropout": 0.5, "n_classes": len(possible_rules)}
    trainer = Trainer(learning_rate=0.05, max_iter=1000, batch_size=1000)
    model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, possible_rules)
    time_bins = np.arange(0, 2.8, 0.1)
    train_accs, test_accs, shuffled_accs, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(model, sess_datas, time_bins, 5, 500, 100, 42)

    base_dir = "/data/patrick_scratch/pseudo"
    np.save(os.path.join(base_dir, f"rule_fb_train_accs.npy"), train_accs)
    np.save(os.path.join(base_dir, f"rule_fb_test_accs.npy"), test_accs)
    np.save(os.path.join(base_dir, f"rule_fb_shuffled_accs.npy"), shuffled_accs)
    np.save(os.path.join(base_dir, f"rule_fb__models.npy"), models)


def main():
    valid_sess = pd.read_pickle("/data/patrick_scratch/multi_sess/valid_sessions.pickle")
    print(f"number of valid sessions: {len(valid_sess)}")
    valid_sess["satisfy"] = valid_sess.apply(check_sess, axis=1)
    valid_sess = valid_sess[valid_sess.satisfy]
    print(f"number of sessions that satisfy criteria: {len(valid_sess)}")
    decode_rule(valid_sess)

if __name__ == "__main__":
    main()