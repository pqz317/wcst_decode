"""
Generate a set of splits ahead of time to use. 
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
from tqdm import tqdm

def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(BeliefPartitionConfigs(), parser)
    args = parser.parse_args()

    for i in tqdm(range(len(FEATURES))):
        args.feat_idx = i
        process_args(args)
        sess_datas = load_session_datas(args)
        all_splits = pd.concat(sess_datas.apply(lambda x: x.get_splits_df()).values)
        all_splits.to_pickle(os.path.join(args.base_output_path, f"splits/{args.feat}_splits.pickle"))



if __name__ == "__main__":
    main()
