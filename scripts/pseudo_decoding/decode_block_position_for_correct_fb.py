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

PRE_INTERVAL = 1300
POST_INTERVAL = 1500
INTERVAL_SIZE = 100
EVENT = "FeedbackOnset"


def create_session_data(sess_name):
    behavior_path = f"/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
    beh = pd.read_csv(behavior_path)
    valid_beh = behavioral_utils.get_valid_trials(beh)
    # find blocks with at least 10 corrects
    valid_beh = valid_beh.groupby("BlockNumber").filter(lambda x: len(x[x.Response == "Correct"]) > 15)
    # session must also have at least 10 blocks like this
    if len(valid_beh.BlockNumber.unique()) < 10: 
        return None
    first_fives = behavioral_utils.get_first_n_corrects_per_block(valid_beh, 5)
    first_fives["label"] = "First"
    last_fives = behavioral_utils.get_last_n_corrects_per_block(valid_beh, 5)
    last_fives["label"] = "Last"
    data = pd.concat((first_fives, last_fives))
    frs = pd.read_pickle(f"/data/patrick_scratch/multi_sess/{sess_name}/{sess_name}_firing_rates_{PRE_INTERVAL}_{EVENT}_{POST_INTERVAL}_{INTERVAL_SIZE}_bins.pickle")
    splitter = ConditionKFoldBlockSplitter(data, condition_column="label", n_splits=5, num_distinct_conditions=2)
    sess_data = SessionData(sess_name, data, frs, splitter)
    sess_data.pre_generate_splits(5)
    return sess_data

def decode_position(valid_sess):
    sess_datas = valid_sess.apply(lambda x: create_session_data(x.session_name), axis=1)
    sess_datas = sess_datas.dropna()
    print(f"{len(sess_datas)} sessions to decode with")
    num_neurons = sess_datas.apply(lambda x: x.get_num_neurons()).sum()
    print(f"{num_neurons} neurons to decode with")
    init_params = {"n_inputs": num_neurons, "p_dropout": 0.5, "n_classes": 2}
    trainer = Trainer(learning_rate=0.05, max_iter=500, batch_size=1000)
    model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, ["First", "Last"])
    time_bins = np.arange(0, 2.8, 0.1)
    train_accs, test_accs, shuffled_accs, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(model, sess_datas, time_bins, 5, 500, 100, 42)

    base_dir = "/data/patrick_scratch/pseudo"
    np.save(os.path.join(base_dir, f"block_position_cor_fb_train_accs.npy"), train_accs)
    np.save(os.path.join(base_dir, f"block_position_cor_fbt_test_accs.npy"), test_accs)
    np.save(os.path.join(base_dir, f"block_position_cor_fb_accs.npy"), shuffled_accs)
    np.save(os.path.join(base_dir, f"block_position_cor_fb_models.npy"), models)


def main():
    valid_sess = pd.read_pickle("/data/patrick_scratch/multi_sess/valid_sessions.pickle")
    decode_position(valid_sess)

if __name__ == "__main__":
    main()