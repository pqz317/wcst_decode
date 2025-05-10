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

import argparse
from belief_partition_configs import add_defaults_to_parser, BeliefPartitionConfigs
import utils.io_utils as io_utils
import json
from decode_belief_partitions import load_session_data, process_args, FEATS_PATH, SESSIONS_PATH
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io

def cross_time_decode(args):
    model_file_name = belief_partitions_io.get_file_name(args)
    output_dir = belief_partitions_io.get_dir_name(args)
    models = np.load(os.path.join(output_dir, f"{model_file_name}_models.npy"), allow_pickle=True)

    # load up session data to train network
    trial_interval = args.trial_interval
    sessions = args.sessions
    sess_datas = sessions.apply(lambda row: load_session_data(
        row, args
    ), axis=1)
    sess_datas = sess_datas.dropna()

    # calculate time bins (in seconds)
    time_bins = np.arange(0, (trial_interval.post_interval + trial_interval.pre_interval) / 1000, trial_interval.interval_size / 1000)
    cross_time_accs = pseudo_classifier_utils.cross_evaluate_by_time_bins(models, sess_datas, time_bins, avg=False)

    save_file_name = belief_partitions_io.get_cross_time_file_name(args)
    np.save(os.path.join(output_dir, f"{save_file_name}_accs.npy"), cross_time_accs)


def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(BeliefPartitionConfigs(), parser)
    args = parser.parse_args()

    process_args(args)
    cross_time_decode(args)


if __name__ == "__main__":
    main()