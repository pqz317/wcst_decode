import s3fs
import os
import scipy.io
import numpy as np
import pandas as pd
import pickle
import torch
import itertools
from . import behavioral_utils
from constants.glm_constants import *

from constants.behavioral_constants import *

HUMAN_LFP_DIR = 'human-lfp'
NHP_DIR = 'nhp-lfp'
NHP_WCST_DIR = 'nhp-lfp/wcst-preprocessed/'

# path for each session, specifying behavior
SESS_BEHAVIOR_PATH = "/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/patrick_res/firing_rates/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_{num_bins_smooth}_smooth.pickle"


def get_fixation_times_path(subject, session):
    return os.path.join(
        NHP_WCST_DIR, "rawdata", f"sub-{subject}", 
        f"sess-{session}", "eye", "itemFixationTimes",
        f"sub-{subject}_sess-{session}_itemFixationTimes.m"
    )


def get_raw_fixation_times(fs, subject, session):
    """Grabs fixation times for each card displayed on the screen
    Note: per trial, each card can have multiple fixations. 

    Args:
        fs: Filesystem to grab from
        subject: str identifier for subject
        sesssion: which session to grab

    Returns:
        np.array of num trials length, with each element as a separate np array
            describing every fixation during the trial 
    """
    fixation_path = get_fixation_times_path(subject, session)
    with fs.open(fixation_path) as fixation_file:
        data = scipy.io.loadmat(fixation_file)
        return data["itemFixationTimes"]

def save_model_outputs(name, interval, split, outputs, base_dir="/data/patrick_scratch/"):
    train_accs_by_bin, test_accs_by_bin, shuffled_accs, models, splits = outputs
    np.save(os.path.join(base_dir, f"{name}_train_accs_{interval}_{split}.npy"), train_accs_by_bin)
    np.save(os.path.join(base_dir, f"{name}_accs_{interval}_{split}.npy"), test_accs_by_bin)
    np.save(os.path.join(base_dir, f"{name}_shuffled_accs_{interval}_{split}.npy"), shuffled_accs)
    np.save(os.path.join(base_dir, f"{name}_models_{interval}_{split}.npy"), models)
    pickle.dump(splits, open(os.path.join(base_dir, f"{name}_splits_{interval}_{split}.npy"), "wb"))

def load_model_outputs(name, interval, split, base_dir="/data/patrick_scratch/"):
    train_accs_by_bin = np.load(os.path.join(base_dir, f"{name}_train_accs_{interval}_{split}.npy"))
    test_accs_by_bin = np.load(os.path.join(base_dir, f"{name}_accs_{interval}_{split}.npy"))
    shuffled_accs = np.load(os.path.join(base_dir, f"{name}_shuffled_accs_{interval}_{split}.npy"))
    models = np.load(os.path.join(base_dir, f"{name}_models_{interval}_{split}.npy"), allow_pickle=True)
    splits = pickle.load(open(os.path.join(base_dir, f"{name}_splits_{interval}_{split}.npy"), "rb"))
    return train_accs_by_bin, test_accs_by_bin, shuffled_accs, models, splits

def load_rpe_sess_beh_and_frs(sess_name, beh_path=SESS_BEHAVIOR_PATH, fr_path=SESS_SPIKES_PATH, set_indices=True, include_prev=False):
    behavior_path = beh_path.format(sess_name=sess_name)
    beh = pd.read_csv(behavior_path)

    # filter trials 
    valid_beh = behavioral_utils.get_valid_trials(beh)
    feature_selections = behavioral_utils.get_selection_features(valid_beh)
    valid_beh = pd.merge(valid_beh, feature_selections, on="TrialNumber")

    # grab the features of the selected card
    valid_beh_rpes = behavioral_utils.get_rpes_per_session(sess_name, valid_beh)
    assert len(valid_beh) == len(valid_beh_rpes)
    pos_med = valid_beh_rpes[valid_beh_rpes.RPE_FE >= 0].RPE_FE.median()
    neg_med = valid_beh_rpes[valid_beh_rpes.RPE_FE < 0].RPE_FE.median()
    # add median labels to 
    def add_group(row):
        rpe = row.RPE_FE
        group = None
        if rpe < neg_med:
            group = "more neg"
        elif rpe >= neg_med and rpe < 0:
            group = "less neg"
        elif rpe >= 0 and rpe < pos_med:
            group = "less pos"
        elif rpe > pos_med:
            group = "more pos"
        row["RPEGroup"] = group
        return row
    valid_beh_rpes = valid_beh_rpes.apply(add_group, axis=1)
    for feature_dim in FEATURE_DIMS:
        valid_beh_rpes[f"{feature_dim}RPEGroup"] = valid_beh_rpes[feature_dim] + "_" + valid_beh_rpes["RPEGroup"]
        valid_beh_rpes[f"{feature_dim}Response"] = valid_beh_rpes[feature_dim] + "_" + valid_beh_rpes["Response"]
    valid_beh_rpes["Card"] = valid_beh_rpes["Color"] + "_" + valid_beh_rpes["Shape"] + "_" + valid_beh_rpes["Pattern"]
    if include_prev:
        columns = FEATURE_DIMS + FEEDBACK_TYPES + ["Card"]
        for feature_dim, fb_type in itertools.product(FEATURE_DIMS, FEEDBACK_TYPES):
            columns.append(f"{feature_dim}{fb_type}")
        for column in columns:
            valid_beh_rpes[f"Prev{column}"] = valid_beh_rpes[column].shift()

    if set_indices:
        valid_beh_rpes = valid_beh_rpes.set_index(["TrialNumber"])

    spikes_path = fr_path.format(
        sess_name=sess_name, 
        pre_interval=PRE_INTERVAL, 
        event=EVENT, 
        post_interval=POST_INTERVAL, 
        interval_size=INTERVAL_SIZE,
        num_bins_smooth=NUM_BINS_SMOOTH,
    )
    frs = pd.read_pickle(spikes_path)
    if set_indices:
        frs = frs.set_index(["TrialNumber"])
    return valid_beh_rpes, frs