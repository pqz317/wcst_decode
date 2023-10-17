import os
import numpy as np
import pandas as pd
import utils.pseudo_utils as pseudo_utils
import utils.pseudo_classifier_utils as pseudo_classifier_utils
import utils.behavioral_utils as behavioral_utils
import utils.spike_utils as spike_utils

from utils.session_data import SessionData

from models.trainer import Trainer
from models.model_wrapper import ModelWrapper, ModelWrapperLinearRegression
from models.multinomial_logistic_regressor import NormedDropoutMultinomialLogisticRegressor

from trial_splitters.rule_condition_block_splitter import RuleConditionBlockSplitter
from trial_splitters.condition_trial_splitter import ConditionTrialSplitter 

PRE_INTERVAL = 1300
POST_INTERVAL = 1500
INTERVAL_SIZE = 50
SMOOTH = 2
EVENT = "FeedbackOnset"

FEATURE_TO_DIM = {
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

# FEATURES = ["SQUARE", "TRIANGLE", "CYAN", "CIRCLE", "YELLOW"]
FEATURES = ["CYAN", "YELLOW", "CIRCLE", "GREEN"]
N_BLOCKS_OF_RULE = 5
SEED = 42
LAST_N_CORRECTS = 3
DECODE_NAME = "is_rule_or_not_last_3_cors"
DATA_MODE = "FiringRate"
WEIGHT_DECAY=0.2



def create_session_data(sess_name, feature):
    behavior_path = f"/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
    beh = pd.read_csv(behavior_path)
    valid_beh = behavioral_utils.get_valid_trials(beh)

    # get last N corrects
    valid_beh = behavioral_utils.get_last_n_corrects_per_block(valid_beh, LAST_N_CORRECTS)
    # label trial with whether the feature is rule or not
    valid_beh["IsRule"] = valid_beh.CurrentRule == feature
    feature_selections = behavioral_utils.get_selection_features(valid_beh)
    valid_beh = pd.merge(valid_beh, feature_selections, on="TrialNumber", how="inner")
    # only look at trials where monkey chose a card containing the feature we're looking at
    feature_dim = FEATURE_TO_DIM[feature]
    valid_beh = valid_beh[valid_beh[feature_dim] == feature]

    # balance the dataset
    is_rule = valid_beh[valid_beh.IsRule]
    not_rule = valid_beh[~valid_beh.IsRule]
    num_trials_per = np.min((len(is_rule), len(not_rule)))
    print(f"{num_trials_per} trials for each condition in session {sess_name}")
    trials = pd.concat([
        is_rule.sample(num_trials_per, random_state=SEED),
        not_rule.sample(num_trials_per, random_state=SEED),
    ])
    fr_path = f"/data/patrick_scratch/multi_sess/{sess_name}/{sess_name}_firing_rates_{PRE_INTERVAL}_{EVENT}_{POST_INTERVAL}_{INTERVAL_SIZE}_bins_{SMOOTH}_smooth.pickle"
    frs = pd.read_pickle(fr_path)

    # filter for just PFC units
    # positions = spike_utils.get_unit_positions_per_sess(sess_name, fr_path)
    # pfc_units = positions[positions.manual_structure == "Prefrontal Cortex"].UnitID.unique()
    # frs = frs[frs.UnitID.isin(pfc_units)]

    # use whichever data mode is specified
    frs = frs.rename(columns={DATA_MODE: "Value"})
    splitter = ConditionTrialSplitter(trials, "IsRule", 0.2)
    return SessionData(sess_name, trials, frs, splitter)

def decode_is_rule_or_not(valid_sess, feature):
    sess_datas = valid_sess.apply(lambda x: create_session_data(x.session_name, feature), axis=1)
    print(f"{len(sess_datas)} sessions meeting criteria")

    num_neurons = sess_datas.apply(lambda x: x.get_num_neurons()).sum()
    print(f"{num_neurons} neurons to decode with")
    init_params = {"n_inputs": num_neurons, "p_dropout": 0.5, "n_classes": 2}
    trainer = Trainer(learning_rate=0.05, max_iter=500, batch_size=1000, weight_decay=WEIGHT_DECAY)
    model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, [True, False])
    interval_secs = INTERVAL_SIZE / 1000
    time_bins = np.arange(0, 2.8, interval_secs)
    train_accs, test_accs, shuffled_accs, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(model, sess_datas, time_bins, 10, 200, 50, SEED)

    base_dir = "/data/patrick_scratch/pseudo"
    np.save(os.path.join(base_dir, f"{feature}_{DECODE_NAME}_{DATA_MODE}_smooth_{SMOOTH}_reg_{WEIGHT_DECAY}_train_accs.npy"), train_accs)
    np.save(os.path.join(base_dir, f"{feature}_{DECODE_NAME}_{DATA_MODE}_smooth_{SMOOTH}_reg_{WEIGHT_DECAY}_test_accs.npy"), test_accs)
    np.save(os.path.join(base_dir, f"{feature}_{DECODE_NAME}_{DATA_MODE}_smooth_{SMOOTH}_reg_{WEIGHT_DECAY}_shuffled_accs.npy"), shuffled_accs)
    np.save(os.path.join(base_dir, f"{feature}_{DECODE_NAME}_{DATA_MODE}_smooth_{SMOOTH}_reg_{WEIGHT_DECAY}_models.npy"), models)


def main():
    for feature in FEATURES:
        print(f"Decoding rule/not for {feature}")
        valid_sess = pd.read_pickle(f"/data/patrick_scratch/multi_sess/valid_sessions_more_than_{N_BLOCKS_OF_RULE}_of_rule.pickle")
        valid_sess = valid_sess[valid_sess[feature]]
        decode_is_rule_or_not(valid_sess, feature)

if __name__ == "__main__":
    main()