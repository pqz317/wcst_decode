"""
Cross-partition generalization: train choice decoder on one belief-dim partition,
evaluate on a different (non-overlapping) partition.

Models are loaded from an existing decode_belief_partitions.py run
(train_partition). Session data is filtered to test_partition trials.
Because the partitions are mutually exclusive, there is no trial leakage.

For shuffle baselines, pass --shuffle_idx N to load the Nth shuffled model
from the train-partition shuffles/ directory and evaluate it on test data.
"""

import os
import copy
import numpy as np
import argparse

import utils.pseudo_classifier_utils as pseudo_classifier_utils
from constants.behavioral_constants import *
from constants.decoding_constants import *

from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import (
    CrossPartitionConfigs,
    add_defaults_to_parser,
)
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
from scripts.pseudo_decoding.belief_partitions.decode_belief_partitions import (
    load_session_datas,
)


def load_train_models(args):
    """Load the trained models from the train-partition run."""
    train_args = copy.deepcopy(args)
    train_args.beh_filters = {"BeliefDimPartition": args.train_partition}
    train_args.balance_by_filters = args.balance_by_filters
    train_args.base_output_path = args.base_model_path

    model_dir = belief_partitions_io.get_dir_name(train_args, make_dir=False)
    file_name = belief_partitions_io.get_file_name(train_args)
    model_path = os.path.join(model_dir, f"{file_name}_models.npy")
    print(f"Loading models from {model_path}", flush=True)
    return np.load(model_path, allow_pickle=True)


def load_test_session_datas(args):
    """Load session data filtered to the test partition."""
    test_args = copy.deepcopy(args)
    test_args.beh_filters = {"BeliefDimPartition": args.test_partition}
    # shuffle_idx must be None here so create_from_splitter uses a deterministic
    # split; the shuffle signal is carried by which *model* we loaded, not the data.
    test_args.shuffle_idx = None
    return load_session_datas(test_args)


def cross_partition_decode(args):
    models = load_train_models(args)

    test_sess_datas = load_test_session_datas(args)

    trial_interval = args.trial_interval
    time_bins = np.arange(
        0,
        (trial_interval.post_interval + trial_interval.pre_interval) / 1000,
        trial_interval.interval_size / 1000,
    )

    accs = pseudo_classifier_utils.evaluate_model_with_data(
        models,
        test_sess_datas,
        time_bins,
        num_train_per_cond=0,
        num_test_per_cond=args.num_test_per_cond,
        condition_label_map=MODE_COND_LABEL_MAPS[args.mode],
    )

    output_dir = belief_partitions_io.get_cross_partition_dir_name(args)
    file_name = belief_partitions_io.get_cross_partition_file_name(args)
    np.save(os.path.join(output_dir, f"{file_name}_accs.npy"), accs)
    print(f"Saved to {output_dir}/{file_name}_accs.npy", flush=True)


def process_args(args):
    args.feat = FEATURES[args.feat_idx]
    args.trial_interval = get_trial_interval(args.trial_event)
    print(
        f"Cross-partition decoding: train={args.train_partition}, "
        f"test={args.test_partition}, feat={args.feat}, mode={args.mode}",
        flush=True,
    )
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(CrossPartitionConfigs(), parser)
    args = parser.parse_args()
    args = process_args(args)
    cross_partition_decode(args)
