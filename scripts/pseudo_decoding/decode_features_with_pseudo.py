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

def load_session_data(sess_name, condition): 
    """
    Loads the data (behavioral and firing rates) for a given session, 
    generates a TrialSplitter based on a condition (feature dimension) 
    creates a SessionData object to store data. 
    Args: 
        sess_name: name of the session to load
        condition: condition used to group trials in pseudo population (in this case a feature dimension)
    Returns: a SessionData object
    """
    behavior_path = f"/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
    beh = pd.read_csv(behavior_path)

    # filter trials 
    valid_beh = beh[beh.Response.isin(["Correct", "Incorrect"])]    

    # grab the features of the selected card
    feature_selections = behavioral_utils.get_selection_features(valid_beh)
    valid_beh_merged = pd.merge(valid_beh, feature_selections, on="TrialNumber", how="inner")

    # load firing rates
    frs = pd.read_pickle(f"/data/patrick_scratch/multi_sess/{sess_name}/{sess_name}_firing_rates_{PRE_INTERVAL}_{EVENT}_{POST_INTERVAL}_{INTERVAL_SIZE}_bins.pickle")

    # create a trial splitter 
    splitter = ConditionTrialSplitter(valid_beh_merged, condition, 0.2)
    return SessionData(sess_name, valid_beh_merged, frs, splitter)

def decode_feature(feature_dim, valid_sess):
    print(f"Decoding {feature_dim}")
    # load all session datas
    sess_datas = valid_sess.apply(lambda x: load_session_data(x.session_name, feature_dim), axis=1)

    # setup decoder, specify all possible label classes, number of neurons, parameters
    classes = possible_features[feature_dim]
    num_neurons = sess_datas.apply(lambda x: x.get_num_neurons()).sum()
    init_params = {"n_inputs": num_neurons, "p_dropout": 0.5, "n_classes": len(classes)}
    # create a trainer object
    trainer = Trainer(learning_rate=0.05, max_iter=1000, batch_size=1000)
    # create a wrapper for the decoder
    model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, classes)
    time_bins = np.arange(0, 2.8, 0.1)
    # train and evaluate the decoder per timein 
    train_accs, test_accs, shuffled_accs, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(model, sess_datas, time_bins, 5, 2000, 400, 42)

    # store the results
    base_dir = "/data/patrick_scratch/pseudo"
    np.save(os.path.join(base_dir, f"{feature_dim}_train_accs.npy"), train_accs)
    np.save(os.path.join(base_dir, f"{feature_dim}_test_accs.npy"), test_accs)
    np.save(os.path.join(base_dir, f"{feature_dim}_shuffled_accs.npy"), shuffled_accs)
    np.save(os.path.join(base_dir, f"{feature_dim}_models.npy"), models)


def main():
    valid_sess = pd.read_pickle("/data/patrick_scratch/multi_sess/valid_sessions.pickle")
    for feature_dim in feature_dims: 
        decode_feature(feature_dim, valid_sess)

if __name__ == "__main__":
    main()