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
import copy
from constants.behavioral_constants import *
from constants.decoding_constants import *

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

def get_ccgp_val_file_name(args):
    # should consist of subject, event, region, next trial value, prev response, 
    pair_str = pair_str = "_".join(args.row.pair)
    shuffle_str = "" if args.shuffle_idx is None else f"_shuffle_{args.shuffle_idx}"
    return f"{pair_str}{shuffle_str}"

def get_ccgp_val_output_dir(args, make_dir=True):
    region_str = "" if args.regions is None else f"_{args.regions.replace(',', '_').replace(' ', '_')}"
    next_trial_str = "_next_trial_value" if args.use_next_trial_value else ""
    prev_response_str = "" if args.prev_response is None else f"_prev_res_{args.prev_response}"
    run_name = f"{args.subject}_{args.trial_interval.event}{region_str}{next_trial_str}{prev_response_str}"
    if args.shuffle_idx is None: 
        dir = os.path.join(args.base_output_path, f"{run_name}")
    else: 
        dir = os.path.join(args.base_output_path, f"{run_name}/shuffles")
    if make_dir: 
        os.makedirs(dir, exist_ok=True)
    return dir

def load_df_from_pairs(args, pairs, dir, shuffle=False):
    res = []
    for i, row in pairs.iterrows():
        # NOTE: hack, need to run 18 runs instead of 17.
        if i < 17:
            args.row = row
            file_name = get_ccgp_val_file_name(args)
            for cond in ["within_cond", "across_cond", "overall"]: 
                acc = np.load(os.path.join(dir, f"{file_name}_{cond}_accs.npy"))
                df = pd.DataFrame(acc).reset_index(names=["Time"])
                ti = args.trial_interval
                df["Time"] = (df["Time"] * ti.interval_size + ti.interval_size - ti.pre_interval) / 1000
                df = df.melt(id_vars="Time", value_vars=list(range(acc.shape[1])), var_name="run", value_name="Accuracy")
                # df["pair"] = feat1, feat2
                df["condition"] = cond if not shuffle else f"{cond}_shuffle"
                res.append(df)
    return pd.concat(res)

def read_ccgp_value(args, pairs, num_shuffles=10):
    """
    Returns two dataframes, one for ccgp one for shuffles
    """
    args.trial_interval = get_trial_interval(args.trial_event)
    dir = get_ccgp_val_output_dir(args, make_dir=False)
    res = load_df_from_pairs(args, pairs, dir)
    for shuffle_idx in range(num_shuffles):
        args.shuffle_idx = shuffle_idx
        dir = get_ccgp_val_output_dir(args, make_dir=False)
        shuffle_res = load_df_from_pairs(args, pairs, dir, shuffle=True)
    res = pd.concat((res, shuffle_res))
    return res

def read_ccgp_value_combine_fb(args, pairs, num_shuffles=10):
    assert args.trial_event == "FeedbackOnset"
    args.trial_interval = get_trial_interval(args.trial_event)

    args.use_next_trial_value = False
    cur_trial_val_res = read_ccgp_value(copy.copy(args), pairs, num_shuffles)

    args.use_next_trial_value = True
    next_trial_val_res = read_ccgp_value(copy.copy(args), pairs, num_shuffles)

    return pd.concat((
        cur_trial_val_res[cur_trial_val_res.Time <=0], 
        next_trial_val_res[next_trial_val_res.Time > 0]
    ))