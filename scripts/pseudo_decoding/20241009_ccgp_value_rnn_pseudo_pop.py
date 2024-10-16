"""
Evaluate CCGP of Value from RNN hidden units, via pseudo populations
"""
import os
import numpy as np
import pandas as pd
import utils.pseudo_utils as pseudo_utils
import utils.pseudo_classifier_utils as pseudo_classifier_utils
import utils.behavioral_utils as behavioral_utils
from constants.behavioral_constants import *
from constants.decoding_constants import *
from utils.session_data import SessionData

from models.trainer import Trainer
from models.model_wrapper import ModelWrapper, ModelWrapperLinearRegression
from models.multinomial_logistic_regressor import NormedDropoutMultinomialLogisticRegressor
from trial_splitters.condition_trial_splitter import ConditionTrialSplitter 
from tqdm import tqdm
import argparse

# the output directory to store the data
OUTPUT_DIR = "/data/patrick_res/rl/pseudo_pop/res/"

# SESSIONS_PATH = "/data/patrick_res/sessions/valid_sessions_rpe.pickle"
# PAIRS_PATH = "/data/patrick_res/sessions/pairs_at_least_3blocks_10sess.pickle"

# path for each session, specifying behavior
SESS_BEHAVIOR_PATH = "/data/patrick_res/rl/pseudo_pop/shared_belief_rnn_prob_matches_sam_0_sess_{sess_name}_beh.csv"
# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/patrick_res/rl/pseudo_pop/shared_belief_rnn_prob_matches_sam_0_sess_{sess_name}_frs.pickle"

DATA_MODE = "FiringRate"

def load_session_data(sess_name, feat):
    behavior_path = SESS_BEHAVIOR_PATH.format(sess_name=sess_name)
    beh = pd.read_csv(behavior_path)
    beh = behavioral_utils.get_belief_value_labels(beh)

    # subselect for either low conf, or high conf preferring feat, where feat is also chosen
    sub_beh = beh[
        (beh.BeliefStateValueLabel == f"High {feat}") |
        (beh.BeliefStateValueLabel == "Low")
    ]
    # balance the conditions out: 
    sub_beh = behavioral_utils.balance_trials_by_condition(sub_beh, ["BeliefStateValueBin"])
    spikes_path = SESS_SPIKES_PATH.format(
        sess_name=sess_name, 
    )
    frs = pd.read_pickle(spikes_path)
    frs = frs.rename(columns={DATA_MODE: "Value"})
    splitter = ConditionTrialSplitter(sub_beh, "BeliefStateValueBin", TEST_RATIO)
    session_data = SessionData(sess_name, sub_beh, frs, splitter)
    session_data.pre_generate_splits(NUM_SPLITS)
    return session_data

def decode():
    within_cond_accs = []
    across_cond_accs = []
    for feat in tqdm(FEATURES):
        print(f"Training decoder for low vs.  high {feat}")
        # load up session data to train network
        sess_names = pd.Series(list(range(16)))
        sess_datas = sess_names.apply(lambda row: load_session_data(row, feat))


        # train the network
        # setup decoder, specify all possible label classes, number of neurons, parameters
        classes = [0, 1]
        num_neurons = sess_datas.apply(lambda x: x.get_num_neurons()).sum()
        init_params = {"n_inputs": num_neurons, "p_dropout": 0.5, "n_classes": len(classes)}
        # create a trainer object
        trainer = Trainer(learning_rate=0.05, max_iter=500, batch_size=1000)
        # create a wrapper for the decoder
        model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, classes)

        # calculate time bins (in seconds)
        time_bins = np.arange(0, 0.1, 0.1)
        train_accs, test_accs, shuffled_accs, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(
            model, sess_datas, time_bins, NUM_SPLITS, NUM_TRAIN_PER_COND, NUM_TEST_PER_COND
        ) 
        within_cond_accs.append(test_accs)

        # next, evaluate network on other dimensions
        other_feats = [f for f in FEATURES if f != feat]
        for other_feat in other_feats: 
            print(f"    Testing decoder on low vs.  high {other_feat}")
            sess_datas = sess_names.apply(lambda row: load_session_data(row, other_feat))
            accs = pseudo_classifier_utils.evaluate_model_with_data(models, sess_datas, time_bins, num_test_per_cond=NUM_TEST_PER_COND)
            across_cond_accs.append(accs)

    within_cond_accs = np.hstack(within_cond_accs)
    across_cond_accs = np.hstack(across_cond_accs)
    print(across_cond_accs.shape)
    np.save(os.path.join(OUTPUT_DIR, f"shared_belief_rnn_prob_matches_sam_0_within_cond_accs.npy"), within_cond_accs)
    np.save(os.path.join(OUTPUT_DIR, f"shared_belief_rnn_prob_matches_sam_0_across_cond_accs.npy"), across_cond_accs)


def main():
    decode()


if __name__ == "__main__":
    main()