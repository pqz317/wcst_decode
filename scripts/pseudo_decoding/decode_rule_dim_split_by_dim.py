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
from sklearn.svm import LinearSVC

PRE_INTERVAL = 1300
POST_INTERVAL = 1500
INTERVAL_SIZE = 100
EVENT = "FeedbackOnset"

RULE_TO_DIM = {
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

FEAUTURE_DIMS = ["COLOR", "SHAPE", "PATTERN"]
SEED = 42
AT_LEAST_N_BLOCKS = 5



def create_session_data(sess_name, feature_dim):
    behavior_path = f"/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
    beh = pd.read_csv(behavior_path)
    valid_beh = behavioral_utils.get_valid_trials(beh)

    # valid_beh["RuleDim"] = valid_beh.apply(lambda x: rule_to_dim[x.CurrentRule], axis=1)
    valid_beh["MatchesDim"] = valid_beh.apply(lambda x: RULE_TO_DIM[x.CurrentRule] == feature_dim, axis=1)
    num_match_blocks = len(valid_beh[valid_beh.MatchesDim].BlockNumber.unique())
    print(f"Session {sess_name} has {num_match_blocks} of rule dim {feature_dim}")
    if num_match_blocks < AT_LEAST_N_BLOCKS:
        print(f"Not enough blocks for {sess_name}, skipping")
        return None
    non_match_blocks = valid_beh[~valid_beh.MatchesDim].BlockNumber.unique()
    rng = np.random.default_rng(SEED)
    subselected_blocks = rng.choice(non_match_blocks, num_match_blocks)
    valid_beh = pd.concat([
        valid_beh[valid_beh.MatchesDim],
        valid_beh[valid_beh.BlockNumber.isin(subselected_blocks)],
    ])
    # trials = behavioral_utils.get_last_n_corrects_per_block(valid_beh, 5)
    # trials = behavioral_utils.get_not_figured_out_trials(valid_beh)
    # just look at correct trials in the blocks
    trials = valid_beh[valid_beh.Response == "Correct"]
    frs = pd.read_pickle(f"/data/patrick_scratch/multi_sess/{sess_name}/{sess_name}_firing_rates_{PRE_INTERVAL}_{EVENT}_{POST_INTERVAL}_{INTERVAL_SIZE}_bins.pickle")

    # splits with 3 distinct conditions, at least 3 blocks for each rule dimension
    # if unable to generate splits, return None
    try: 
        splitter = RuleConditionBlockSplitter(trials, condition="MatchesDim", num_distinct_conditions=2, num_blocks_per_cond=5)
    except ValueError:
        return None

    return SessionData(sess_name, trials, frs, splitter)

def decode_rule_dim(valid_sess, feature_dim):
    print(f"Decoding for whether rule dim matches {feature_dim}")
    sess_datas = valid_sess.apply(lambda x: create_session_data(x.session_name, feature_dim), axis=1)
    print(f"{len(sess_datas)} total sessions considered")
    sess_datas = sess_datas.dropna()
    print(f"{len(sess_datas)} sessions meeting criteria")
    num_neurons = sess_datas.apply(lambda x: x.get_num_neurons()).sum()
    print(f"{num_neurons} neurons to decode with")
    # init_params = {"n_inputs": num_neurons, "p_dropout": 0.5, "n_classes": 2}
    # trainer = Trainer(learning_rate=0.05, max_iter=1000, batch_size=1000)
    # model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, [True, False])
    model = LinearSVC(max_iter=5000)
    time_bins = np.arange(0, 2.8, 0.1)
    train_accs, test_accs, shuffled_accs, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(model, sess_datas, time_bins, 5, 2000, 400, SEED)

    base_dir = "/data/patrick_scratch/pseudo"
    np.save(os.path.join(base_dir, f"rule_dim_matches_{feature_dim}_cor_only_svm_train_accs.npy"), train_accs)
    np.save(os.path.join(base_dir, f"rule_dim_matches_{feature_dim}_cor_only_svm_test_accs.npy"), test_accs)
    np.save(os.path.join(base_dir, f"rule_dim_matches_{feature_dim}_cor_only_svm_shuffled_accs.npy"), shuffled_accs)
    np.save(os.path.join(base_dir, f"rule_dim_matches_{feature_dim}_cor_only_svm_models.npy"), models)


def main():
    for dim in FEAUTURE_DIMS:
        valid_sess = pd.read_pickle("/data/patrick_scratch/multi_sess/valid_sessions.pickle")
        decode_rule_dim(valid_sess, dim)

if __name__ == "__main__":
    main()