import os
import numpy as np
import utils.pseudo_classifier_utils as pseudo_classifier_utils

from constants.behavioral_constants import *
from constants.decoding_constants import *

import argparse
from belief_partition_configs import add_defaults_to_parser, BeliefPartitionConfigs

from decode_belief_partitions import load_session_datas, process_args, FEATS_PATH, SESSIONS_PATH
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
import copy
import pandas as pd

"""
An attempt at using separate decoders for choice, reward to decode interaction of choice/reward, eg chose AND correct vs. Not. 
"""

import os
import numpy as np
import utils.pseudo_classifier_utils as pseudo_classifier_utils

from constants.behavioral_constants import *
from constants.decoding_constants import *

import argparse
from belief_partition_configs import add_defaults_to_parser, BeliefPartitionConfigs

from decode_belief_partitions import load_session_datas, process_args, FEATS_PATH, SESSIONS_PATH
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
import copy
import pandas as pd
from models.choice_reward_model import create_models

def load_mode_models(args, mode):
    model_args = copy.deepcopy(args)
    model_args.mode = mode
    model_file_name = belief_partitions_io.get_file_name(model_args)
    model_dir = belief_partitions_io.get_dir_name(model_args)
    models = np.load(os.path.join(model_dir, f"{model_file_name}_models.npy"), allow_pickle=True)
    return models

def choice_reward_decode(args):
    choice_models = load_mode_models(args, "choice_int")
    reward_models = load_mode_models(args, "reward_int")

    choice_reward_models = create_models(choice_models, reward_models, args.mode)
    splits_df = pd.read_pickle(os.path.join(args.base_output_path, f"splits/{args.feat}_splits.pickle"))

    # load up session data to train network
    trial_interval = args.trial_interval
    sess_datas = load_session_datas(args, splits_df)

    # calculate time bins (in seconds)
    time_bins = np.arange(0, (trial_interval.post_interval + trial_interval.pre_interval) / 1000, trial_interval.interval_size / 1000)
    accs = pseudo_classifier_utils.evaluate_model_with_data(choice_reward_models, sess_datas, time_bins, condition_label_map=MODE_COND_LABEL_MAPS[args.mode])

    save_args = copy.deepcopy(args)
    save_args.mode = f"{args.mode}_separate"
    save_file_name = belief_partitions_io.get_file_name(save_args)
    save_dir = belief_partitions_io.get_dir_name(save_args)
    np.save(os.path.join(save_dir, f"{save_file_name}_test_accs.npy"), accs)


def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(BeliefPartitionConfigs(), parser)
    args = parser.parse_args()
    for feat_idx in range(12):
        print(f"Calculating for feat {FEATURES[feat_idx]}")
        args.feat_idx = feat_idx
        process_args(args)
        choice_reward_decode(args)

if __name__ == "__main__":
    main()