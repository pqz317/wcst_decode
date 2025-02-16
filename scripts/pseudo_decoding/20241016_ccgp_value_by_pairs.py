"""
Evaluate CCGP of Value in neural population, computing belief state value from belief state agent
"""
import os
import numpy as np
import pandas as pd
import utils.pseudo_utils as pseudo_utils
import utils.pseudo_classifier_utils as pseudo_classifier_utils
import utils.behavioral_utils as behavioral_utils
import utils.spike_utils as spike_utils

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
OUTPUT_DIR = "/data/res/pseudo"
# path to a dataframe of sessions to analyze
SESSIONS_PATH = "/data/patrick_res/sessions/{sub}/valid_sessions.pickle"
SA_PAIRS_PATH = "/data/patrick_res/sessions/SA/pairs_at_least_3blocks_7sess.pickle"
SA_MORE_SESS_PAIRS_PATH = "/data/patrick_res/sessions/SA/pairs_at_least_3blocks_10sess.pickle"
# BL_PAIRS_PATH = "/data/patrick_res/sessions/BL/pairs_at_least_1blocks_3sess.pickle"
BL_PAIRS_PATH = "/data/patrick_res/sessions/BL/pairs_at_least_2blocks_1sess.pickle"

# path for each session, specifying behavior
SESS_BEHAVIOR_PATH = "/data/patrick_res/behavior/{sub}/{sess_name}_object_features.csv"
# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/patrick_res/firing_rates/{sub}/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"
SIMULATED_SPIKES_PATH = "/data/patrick_res/firing_rates/{sess_name}_firing_rates_simulated_noise_{noise}.pickle"

DATA_MODE = "FiringRate"

REGIONS = ["anterior", "temporal", None]
UNITS_PATH = "/data/patrick_res/firing_rates/{sub}/all_units.pickle"

def load_session_data(row, cond, region_units, args):
    """
    cond: either a feature or a pair of features: 
    """
    sess_name = row.session_name
    behavior_path = SESS_BEHAVIOR_PATH.format(
        sess_name=sess_name, 
        sub=args.subject,
    )
    beh = pd.read_csv(behavior_path)
    beh = behavioral_utils.get_valid_trials(beh, args.subject)
    feature_selections = behavioral_utils.get_selection_features(beh)
    beh = pd.merge(beh, feature_selections, on="TrialNumber", how="inner")
    beh = behavioral_utils.get_beliefs_per_session(beh, sess_name, args.subject)
    if args.use_next_trial_value:
        beh = behavioral_utils.shift_beliefs(beh)

    if args.prev_response is not None: 
        beh["PrevResponse"] = beh.Response if args.use_next_trial_value else beh.Response.shift()
        beh = beh[~beh.PrevResponse.isna()]
        beh = behavioral_utils.balance_trials_by_condition(beh, ["PrevResponse"])
        beh = beh[beh.PrevResponse == args.prev_response]

    beh = behavioral_utils.get_belief_value_labels(beh)

    if args.shuffle_idx is not None: 
        beh = behavioral_utils.shuffle_beh_by_shift(beh, buffer=50, seed=args.shuffle_idx)


    # subselect for either low conf, or high conf preferring feat, where feat is also chosen
    if len(cond) == 1:
        feat = cond[0] 
        sub_beh = beh[
            ((beh[FEATURE_TO_DIM[feat]] == feat) & (beh.BeliefStateValueLabel == f"High {feat}")) |
            (beh.BeliefStateValueLabel == "Low")
        ]
    elif len(cond) == 2: 
        feat1, feat2 = cond
        sub_beh = beh[
            ((beh[FEATURE_TO_DIM[feat1]] == feat1) & (beh.BeliefStateValueLabel == f"High {feat1}")) |
            ((beh[FEATURE_TO_DIM[feat2]] == feat2) & (beh.BeliefStateValueLabel == f"High {feat2}")) |
            (beh.BeliefStateValueLabel == "Low")
        ]
    else: 
        raise ValueError("cond must be either 1 or 2 elements")



    # balance the conditions out: 
    sub_beh = behavioral_utils.balance_trials_by_condition(sub_beh, ["BeliefStateValueBin"])
    trial_interval = args.trial_interval
    spikes_path = SESS_SPIKES_PATH.format(
        sub=args.subject,
        sess_name=sess_name, 
        pre_interval=trial_interval.pre_interval, 
        event=trial_interval.event, 
        post_interval=trial_interval.post_interval, 
        interval_size=trial_interval.interval_size
    )
    print(spikes_path)
    frs = pd.read_pickle(spikes_path)
    frs = frs.rename(columns={DATA_MODE: "Value"})
    if region_units is not None: 
        frs["PseudoUnitID"] = int(sess_name) * 100 + frs.UnitID.astype(int)
        frs = frs[frs.PseudoUnitID.isin(region_units)]
    splitter = ConditionTrialSplitter(sub_beh, "BeliefStateValueBin", TEST_RATIO)
    session_data = SessionData(sess_name, sub_beh, frs, splitter)
    session_data.pre_generate_splits(NUM_SPLITS)
    return session_data


def train_decoder(sess_datas, time_bins):
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
    train_accs, test_accs, shuffled_accs, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(
        model, sess_datas, time_bins, NUM_SPLITS, NUM_TRAIN_PER_COND, NUM_TEST_PER_COND
    ) 
    return train_accs, test_accs, shuffled_accs, models

def get_name(args):
    pair_str = "_".join(args.row.pair)
    region_str = "" if args.region is None else f"_{args.region}"
    next_trial_str = "_next_trial_value" if args.use_next_trial_value else ""
    prev_response_str = "" if args.prev_response is None else f"_prev_res_{args.prev_response}"
    shuffle_str = "" if args.shuffle_idx is None else f"_shuffle_{args.shuffle_idx}"
    more_sess_str = "_more_sess" if args.more_sess else ""
    name = f"{args.subject}_ccgp_belief_state_value_{args.trial_interval.event}_pair_{pair_str}{region_str}{next_trial_str}{prev_response_str}{more_sess_str}{shuffle_str}"
    return name

def create_shuffle_dir(name):
    shuffle_dir = os.path.join(OUTPUT_DIR, f"{name}_shuffles")
    os.makedirs(shuffle_dir, exist_ok=True)
    return shuffle_dir

    
def decode(args):
    region_units = spike_utils.get_region_units(args.region, UNITS_PATH.format(sub=args.subject))
    trial_interval = args.trial_interval
    sessions = args.sessions
    name = get_name(args)
    output_dir = OUTPUT_DIR if args.shuffle_idx is None else create_shuffle_dir(name)

    pair = args.row.pair
    within_cond_accs = []
    across_cond_accs = []
    for feat in pair: 
        print(f"Training decoder for low vs.  high {feat}")
        # load up session data to train network
        train_feat_sess_datas = args.sessions.apply(lambda row: load_session_data(row, [feat], region_units, args), axis=1)

        time_bins = np.arange(0, (trial_interval.post_interval + trial_interval.pre_interval) / 1000, trial_interval.interval_size / 1000)
        train_accs, test_accs, shuffled_accs, models = train_decoder(train_feat_sess_datas, time_bins)
        within_cond_accs.append(test_accs)

        # next, evaluate network on other dimensions
        test_feat = [f for f in pair if f != feat][0]
        test_feat_sess_datas = sessions.apply(lambda row: load_session_data(row, [test_feat], region_units, args), axis=1)
        accs_across_time = pseudo_classifier_utils.evaluate_model_with_data(models, test_feat_sess_datas, time_bins, num_test_per_cond=NUM_TEST_PER_COND)
        across_cond_accs.append(accs_across_time)
        np.save(os.path.join(output_dir, f"{name}_feat_{feat}_models.npy"), models)
    within_cond_accs = np.hstack(within_cond_accs)
    across_cond_accs = np.hstack(across_cond_accs)

    overall_sess_datas = sessions.apply(lambda row: load_session_data(row, pair, region_units, args), axis=1)
    train_accs, test_accs, shuffled_accs, models = train_decoder(overall_sess_datas, time_bins)
    np.save(os.path.join(output_dir, f"{name}_overall_accs.npy"), test_accs)
    np.save(os.path.join(output_dir, f"{name}_overall_models.npy"), models)

    np.save(os.path.join(output_dir, f"{name}_within_cond_accs.npy"), within_cond_accs)
    np.save(os.path.join(output_dir, f"{name}_across_cond_accs.npy"), across_cond_accs)

def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair_idx', type=int)
    parser.add_argument('--region_idx', default=None, type=int)
    parser.add_argument('--subject', default="SA", type=str)
    parser.add_argument('--trial_event', default="StimOnset", type=str)
    parser.add_argument('--use_next_trial_value', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--prev_response', default=None, type=str)
    parser.add_argument('--shuffle_idx', default=None, type=int)
    parser.add_argument('--more_sess', action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()
    subject = args.subject
    if subject == "SA": 
        if args.more_sess:
            pairs = pd.read_pickle(SA_MORE_SESS_PAIRS_PATH)
        else: 
            pairs = pd.read_pickle(SA_PAIRS_PATH)
    else: 
        pairs = pd.read_pickle(BL_PAIRS_PATH)
    args.row = pairs.iloc[args.pair_idx]
    valid_sess = pd.read_pickle(SESSIONS_PATH.format(sub=subject))
    args.sessions = valid_sess[valid_sess.session_name.isin(args.row.sessions)]
    args.region =  None if args.region_idx is None else REGIONS[args.region_idx]
    args.trial_interval = get_trial_interval(args.trial_event)

    print(f"Computing CCGP for {subject} of belief state value in interval {args.trial_event}", flush=True)
    if args.more_sess:
        print(f"Using more sessions", flush=True)
    print(f"shuffle idx is {args.shuffle_idx}", flush=True)
    print(f"Looking at region {args.region}, using use_next_trial_value {args.use_next_trial_value}", flush=True)
    print(f"examining conditions between {args.row.pair} using between {args.row.num_sessions} sessions", flush=True)
    print(f"Conditioning on prev response being {args.prev_response}", flush=True)
    decode(args)

if __name__ == "__main__":
    main()