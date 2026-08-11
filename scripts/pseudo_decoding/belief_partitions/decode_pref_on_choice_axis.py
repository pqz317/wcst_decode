"""
Script for projecting preference activity onto the choice decoder's axis, a feature at a time.

Preference decoding asks how separable High X vs. High Not X activity is, in trials where feature
X was chosen and correct. This asks how separable those same two conditions are once projected
onto the choice axis (Chose X vs. Not Chose X, fit on all units), to see how much of preference
decodability the choice axis already captures.

Uses the same pseudo population generation and train/test splits as preference decoding. The axis
direction is never refit: only a threshold along it is, with the sign fixed so that High X sits on
the Chose side of the axis.

Which choice run the axis comes from is set by --axis_beh_filters, the trials that run was fit on:
{} for the all-trials runs, {"Response": "Correct"} for the correct-only re-runs. The filters are
part of the mode results are stored under, so runs against different axes sit side by side.
"""

import os
import numpy as np
import pandas as pd
import utils.pseudo_classifier_utils as pseudo_classifier_utils

from constants.behavioral_constants import *
from constants.decoding_constants import *

import argparse
import json
from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import BeliefPartitionConfigs, add_defaults_to_parser
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
import scripts.pseudo_decoding.belief_partitions.decode_belief_partitions as decode_belief_partitions
import copy

# mode supplying the trials, splits, and conditions being separated
PREF_MODE = "pref"
# mode supplying the axis they're projected onto, read from all-unit choice runs
AXIS_MODE = "choice"
CHOICE_AXIS_PATH = "/data/patrick_res/choice_reward"
# mode results are stored under, so they read back alongside preference decoding
PROJ_MODE = "pref_on_choice"
PROJ_OUTPUT_PATH = "/data/patrick_res/choice_axis_projection_accs"


def get_proj_mode(axis_beh_filters):
    """
    Mode results are stored under. The axis run's filters are part of the name, so projections
    onto different choice axes sit side by side rather than overwriting each other: an axis fit
    on all trials stays "pref_on_choice", one fit on correct trials only becomes
    "pref_on_choice_Response_Correct".
    """
    filt_str = belief_partitions_io.get_filter_str(axis_beh_filters)
    return f"{PROJ_MODE}_{filt_str}" if filt_str else PROJ_MODE


def load_choice_axes(args, num_bins):
    """
    Loads the choice decoder's axis for every time bin, averaged over the choice run's splits.

    Each split's axis is taken in raw firing rate units first, weights / batch norm std, then
    averaged, rather than averaging weights and stds separately.
    Returns the axes as num_bins x num_units, along with the ascending PseudoUnitIDs of the
    columns.
    """
    axis_args = copy.deepcopy(args)
    axis_args.mode = AXIS_MODE
    # the all units choice runs use no subpopulation, and only whichever trials they were fit on
    axis_args.beh_filters = args.axis_beh_filters
    axis_args.sig_unit_level = None
    axis_args.shuffle_idx = None
    axis_args.base_output_path = CHOICE_AXIS_PATH

    axis_dir = belief_partitions_io.get_dir_name(axis_args, make_dir=False)
    print(f"Reading choice axes from {axis_dir}", flush=True)
    models = np.load(
        os.path.join(axis_dir, f"{belief_partitions_io.get_file_name(axis_args)}_models.npy"),
        allow_pickle=True
    )
    # read_units sorts by PseudoUnitID, matching the column order of coef_
    units = belief_partitions_io.read_units(axis_args, [args.feat])

    classes = MODE_TO_CLASSES[AXIS_MODE]
    if models[0, 0].idx_to_labels != dict(enumerate(classes)):
        raise ValueError(f"choice models have labels {models[0, 0].idx_to_labels}, expected {classes}")
    if models.shape[0] != num_bins:
        raise ValueError(f"choice models have {models.shape[0]} time bins, expected {num_bins}")
    if models[0, 0].coef_.shape[1] != len(units):
        raise ValueError(f"choice models have {models[0, 0].coef_.shape[1]} units, expected {len(units)}")

    high_idx = classes.index(MODE_TO_DIRECTION_LABELS[AXIS_MODE]["high"])
    low_idx = classes.index(MODE_TO_DIRECTION_LABELS[AXIS_MODE]["low"])

    axes = np.empty((num_bins, len(units)))
    for bin_idx in range(num_bins):
        per_split = []
        for model in models[bin_idx, :]:
            weights_diff = model.coef_[high_idx, :] - model.coef_[low_idx, :]
            # 1e-5 from torch batchnorm1d, numerical
            std = np.sqrt(model.model.norm.running_var.detach().cpu().numpy() + 1e-5)
            per_split.append(weights_diff / std)
        axes[bin_idx, :] = np.mean(np.vstack(per_split), axis=0)
    return axes, units.PseudoUnitID.to_numpy()


def evaluate_projections(sess_datas, axes, axis_unit_ids, time_bins, args):
    """
    For every time bin and split, generates pseudo trials the same way preference decoding does,
    projects them onto the choice axis, refits a threshold on train trials, scores on test ones.
    """
    pref_unit_ids = np.concatenate(sess_datas.apply(lambda x: x.get_pseudo_unit_ids()).values)
    # the choice runs cover all units, so this should drop very few preference units
    shared_unit_ids = np.sort(np.intersect1d(pref_unit_ids, axis_unit_ids))
    print(f"projecting {len(shared_unit_ids)} of {len(pref_unit_ids)} units onto the choice axis", flush=True)
    # align axis columns to the ascending unit order transform_input_data produces
    axes = axes[:, np.searchsorted(axis_unit_ids, shared_unit_ids)]

    high_label = MODE_TO_DIRECTION_LABELS[PREF_MODE]["high"]
    test_accs = np.empty((len(time_bins), args.num_splits))
    for bin_idx, time_bin in enumerate(time_bins):
        print(f"Working on bin {time_bin}", flush=True)
        for split_idx in range(args.num_splits):
            pseudo_sess = pd.concat(sess_datas.apply(
                lambda x: x.generate_pseudo_data(args.num_train_per_cond, args.num_test_per_cond, time_bin, split_idx)
            ).values, ignore_index=True)
            pseudo_sess = pseudo_sess[pseudo_sess.PseudoUnitID.isin(shared_unit_ids)]

            train_data = pseudo_sess[pseudo_sess.Type == "Train"]
            test_data = pseudo_sess[pseudo_sess.Type == "Test"]

            proj_train = pseudo_classifier_utils.transform_input_data(train_data) @ axes[bin_idx, :]
            proj_test = pseudo_classifier_utils.transform_input_data(test_data) @ axes[bin_idx, :]
            train_pos = pseudo_classifier_utils.transform_label_data(train_data) == high_label
            test_pos = pseudo_classifier_utils.transform_label_data(test_data) == high_label

            threshold = pseudo_classifier_utils.fit_threshold(proj_train, train_pos)
            test_accs[bin_idx, split_idx] = pseudo_classifier_utils.score_threshold(proj_test, test_pos, threshold)
    return test_accs


def project(args):
    splits = None
    if args.shuffle_idx is None:
        # reuse the exact splits the preference run used.
        # shuffles regenerate their own, since their conditions differ
        pref_dir = belief_partitions_io.get_dir_name(args, make_dir=False)
        pref_name = belief_partitions_io.get_file_name(args)
        splits = pd.read_pickle(os.path.join(pref_dir, f"{pref_name}_splits.pickle"))
    sess_datas = decode_belief_partitions.load_session_datas(args, splits_df=splits)

    # calculate time bins (in seconds), same as decode_belief_partitions
    trial_interval = args.trial_interval
    time_bins = np.arange(0, (trial_interval.post_interval + trial_interval.pre_interval) / 1000, trial_interval.interval_size / 1000)

    axes, axis_unit_ids = load_choice_axes(args, len(time_bins))
    test_accs = evaluate_projections(sess_datas, axes, axis_unit_ids, time_bins, args)

    # store under the projection's own mode and path, everything else named as the preference run
    save_args = copy.deepcopy(args)
    save_args.mode = get_proj_mode(args.axis_beh_filters)
    save_args.base_output_path = PROJ_OUTPUT_PATH
    output_dir = belief_partitions_io.get_dir_name(save_args)
    file_name = belief_partitions_io.get_file_name(save_args)
    np.save(os.path.join(output_dir, f"{file_name}_test_accs.npy"), test_accs)


def process_args(args):
    """
    Determines feature, trial interval to use, adds them to args
    """
    # trials, splits and conditions all come from the preference run
    args.mode = PREF_MODE
    args.feat = FEATURES[args.feat_idx]
    args.trial_interval = get_trial_interval(args.trial_event)
    print(f"Projecting {args.mode} activity for feat {args.feat} onto the {AXIS_MODE} axis", flush=True)
    print(f"With filters {args.beh_filters}, onto an axis fit with filters {args.axis_beh_filters}", flush=True)
    print(f"Storing results under mode {get_proj_mode(args.axis_beh_filters)}", flush=True)
    if args.sig_unit_level:
        print(f"Using only units that are selective with signifance level {args.sig_unit_level}")
    return args


def main(args):
    args = process_args(args)
    project(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(BeliefPartitionConfigs(), parser)
    # not part of BeliefPartitionConfigs, that's shared by every decoding script and only this one
    # reads a second run's axis. Parsed as json, same as --beh_filters
    parser.add_argument('--axis_beh_filters', default={}, type=lambda x: json.loads(x))
    args = parser.parse_args()
    main(args)
