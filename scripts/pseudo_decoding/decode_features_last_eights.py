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
from trial_splitters.condition_trial_splitter import ConditionTrialSplitter

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

split_seed = 42


def create_session_data(sess_name, splitter_name, feature_dim):
    behavior_path = f"/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
    beh = pd.read_csv(behavior_path)
    valid_beh = behavioral_utils.get_valid_trials(beh)
    last_eights = behavioral_utils.get_last_n_corrects_per_block(valid_beh, 8)
    feature_selections = behavioral_utils.get_selection_features(last_eights)
    last_eights_merged = pd.merge(last_eights, feature_selections, on="TrialNumber", how="inner")
    print(sess_name)
    # trials = behavioral_utils.get_last_n_corrects_per_block(valid_beh, 5)
    frs = pd.read_pickle(f"/data/patrick_scratch/multi_sess/{sess_name}/{sess_name}_firing_rates_{PRE_INTERVAL}_{EVENT}_{POST_INTERVAL}_{INTERVAL_SIZE}_bins.pickle")
    if splitter_name == "random":
        splitter = ConditionTrialSplitter(last_eights_merged, feature_dim, 0.2)
    else:
        splitter = ConditionKFoldBlockSplitter(last_eights_merged, feature_dim, 5, seed=split_seed)
    session_data =  SessionData(sess_name, last_eights_merged, frs, splitter)
    try: 
        session_data.pre_generate_splits(5)
        return session_data
    except ValueError as e:
        print(f"Error while generating splits for session {sess_name}, skipping:")
        print(e)
        return None

def decode_feature(feature_dim, valid_sess, splitter_name):
    print(f"Decoding {feature_dim} with splitter {splitter_name}")
    sess_datas = valid_sess.apply(lambda x: create_session_data(x.session_name, splitter_name, feature_dim), axis=1)
    sess_datas = sess_datas.dropna()
    print(f"{len(sess_datas)} sessions successfully generated splits, using them")
    classes = possible_features[feature_dim]
    num_neurons = sess_datas.apply(lambda x: x.get_num_neurons()).sum()
    init_params = {"n_inputs": num_neurons, "p_dropout": 0.5, "n_classes": len(classes)}
    trainer = Trainer(learning_rate=0.05, max_iter=1000, batch_size=1000)
    model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, classes)
    time_bins = np.arange(0, 2.8, 0.1)
    train_accs, test_accs, shuffled_accs, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(model, sess_datas, time_bins, 5, 2000, 400, 42)

    base_dir = "/data/patrick_scratch/pseudo"
    np.save(os.path.join(base_dir, f"last_eights_{feature_dim}_{splitter_name}_train_accs.npy"), train_accs)
    np.save(os.path.join(base_dir, f"last_eights_{feature_dim}_{splitter_name}_test_accs.npy"), test_accs)
    np.save(os.path.join(base_dir, f"last_eights_{feature_dim}_{splitter_name}_shuffled_accs.npy"), shuffled_accs)
    np.save(os.path.join(base_dir, f"last_eights_{feature_dim}_{splitter_name}_models.npy"), models)


def main():
    valid_sess = pd.read_pickle("/data/patrick_scratch/multi_sess/valid_sessions.pickle")
    for feature_dim in feature_dims: 
        decode_feature(feature_dim, valid_sess, "random")
        decode_feature(feature_dim, valid_sess, "block")

if __name__ == "__main__":
    main()