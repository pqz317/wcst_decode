import os
import numpy as np
import utils.pseudo_classifier_utils as pseudo_classifier_utils
import pandas as pd

from constants.behavioral_constants import *
from constants.decoding_constants import *

import argparse
from belief_partition_configs import add_defaults_to_parser, BeliefPartitionConfigs

from decode_belief_partitions import load_session_datas, train_decoder, SESSIONS_PATH
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io

SA_MORE_SESS_PAIRS_PATH = "/data/patrick_res/sessions/SA/pairs_at_least_3blocks_10sess_more_sess.pickle"
# BL_PAIRS_PATH = "/data/patrick_res/sessions/BL/pairs_at_least_1blocks_3sess.pickle"
BL_PAIRS_PATH = "/data/patrick_res/sessions/BL/pairs_at_least_2blocks_6sess.pickle"

def ccgp(args):
    file_name = belief_partitions_io.get_ccgp_file_name(args)
    output_dir = belief_partitions_io.get_dir_name(args)

    pair = args.feat_pair

    within_cond_accs = []
    across_cond_accs = []
    for feat in pair: 
        print(f"Training decoder preference in {feat}")
        # load up session data to train network
        args.feat = feat
        train_feat_sess_datas = load_session_datas(args)
        test_accs, models = train_decoder(train_feat_sess_datas, args)
        within_cond_accs.append(test_accs)

        # next, evaluate network on other dimensions
        test_feat = [f for f in pair if f != feat][0]
        args.feat = test_feat
        test_feat_sess_datas = load_session_datas(args)
        trial_interval = args.trial_interval
        time_bins = np.arange(0, (trial_interval.post_interval + trial_interval.pre_interval) / 1000, trial_interval.interval_size / 1000)
        accs_across_time = pseudo_classifier_utils.evaluate_model_with_data(models, test_feat_sess_datas, time_bins, num_test_per_cond=NUM_TEST_PER_COND)
        across_cond_accs.append(accs_across_time)
        np.save(os.path.join(output_dir, f"{file_name}_feat_{feat}_models.npy"), models)
    within_cond_accs = np.hstack(within_cond_accs)
    across_cond_accs = np.hstack(across_cond_accs)

    np.save(os.path.join(output_dir, f"{file_name}_within_cond_accs.npy"), within_cond_accs)
    np.save(os.path.join(output_dir, f"{file_name}_across_cond_accs.npy"), across_cond_accs)


def process_args(args):
    """
    Determines features, sessions, trial intervals, to use for decoding,
    Adds them to args
    """
    valid_sess = pd.read_pickle(SESSIONS_PATH.format(sub=args.subject))

    subject = args.subject
    if subject == "SA": 
        pairs = pd.read_pickle(SA_MORE_SESS_PAIRS_PATH)
    else: 
        pairs = pd.read_pickle(BL_PAIRS_PATH)
    pair_row = pairs.iloc[args.pair_idx]
    args.feat_pair = pair_row.pair
    
    args.sessions = valid_sess[valid_sess.session_name.isin(pair_row.sessions)]
    args.trial_interval = get_trial_interval(args.trial_event)

    print(args.beh_filters)
    print(f"Cross decoding for across pair {args.feat_pair} using {len(args.sessions)} sessions, mode {args.mode}", flush=True)
    print(f"Using {args.fr_type} as inputs", flush=True)
    print(f"With filters {args.beh_filters}", flush=True)
    if args.use_v2_pseudo: 
        print(f"Using new pseudo data generation")
    if args.sig_unit_level:
        print(f"Using only units that are selective with signifance level {args.sig_unit_level}")
    return args


def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(BeliefPartitionConfigs(), parser)
    args = parser.parse_args()

    process_args(args)
    ccgp(args)

if __name__ == "__main__":
    main()