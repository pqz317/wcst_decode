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
SESS_SPIKES_PATH = "/data/patrick_res/firing_rates/{sub}/{sess_name}_{fr_type}_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"


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
    pair_str = pair_str = "_".join(args.feat_pair)
    shuffle_str = "" if args.shuffle_idx is None else f"_shuffle_{args.shuffle_idx}"
    return f"{pair_str}{shuffle_str}"    

def get_ccgp_val_output_dir(args, make_dir=True):
    region_str = "" if args.regions is None else f"_{args.regions.replace(',', '_').replace(' ', '_')}"
    next_trial_str = "_next_trial_value" if args.use_next_trial_value else ""
    prev_response_str = "" if args.prev_response is None else f"_prev_res_{args.prev_response}"
    fr_type_str = f"_{args.fr_type}" if args.fr_type != "firing_rates" else ""
    filt_str = "".join([f"_{k}_{v}"for k, v in args.beh_filters.items()])
    sig_units_str = f"_{args.sig_unit_level}_units" if args.sig_unit_level else ""

    run_name = f"{args.subject}_{args.trial_event}{fr_type_str}{region_str}{next_trial_str}{prev_response_str}{filt_str}{sig_units_str}"
    if args.shuffle_idx is None: 
        dir = os.path.join(args.base_output_path, f"{run_name}")
    else:
        if args.shuffle_method:  
            dir = os.path.join(args.base_output_path, f"{run_name}/{args.shuffle_method}_shuffles")
        else: 
            # backwards compatibility
            dir = os.path.join(args.base_output_path, f"{run_name}/shuffles")
    if make_dir: 
        os.makedirs(dir, exist_ok=True)
    return dir


def get_preferred_beliefs_file_name(args):
    """
    Naming convention for preferred beliefs decoding files
    """
    pair = args.feat_pair
    not_pref_str = "chosen_not_pref_" if args.chosen_not_preferred else ""
    pair_str = "_".join(pair)
    shuffle_str = "" if args.shuffle_idx is None else f"_shuffle_{args.shuffle_idx}"
    return f"{not_pref_str}{pair_str}{shuffle_str}"

def get_preferred_beliefs_output_dir(args, make_dir=True):
    """
    Directory convention for preferred beliefs decoding
    """
    region_str = "" if args.regions is None else f"_{args.regions.replace(',', '_').replace(' ', '_')}"
    fr_type_str = f"_{args.fr_type}" if args.fr_type != "firing_rates" else ""
    filt_str = "".join([f"_{k}_{v}"for k, v in args.beh_filters.items()])
    sig_units_str = f"_{args.sig_unit_level}_units" if args.sig_unit_level else ""

    run_name = f"{args.subject}_{args.trial_event}{fr_type_str}{region_str}{filt_str}{sig_units_str}"
    if args.shuffle_idx is None: 
        dir = os.path.join(args.base_output_path, f"{run_name}")
    else: 
        dir = os.path.join(args.base_output_path, f"{run_name}/shuffles")
    if make_dir: 
        os.makedirs(dir, exist_ok=True)
    return dir

def get_selected_features_file_name(args, cond=None):
    """
    Naming convention for preferred beliefs decoding files
    """
    if cond: 
        args.condition = cond
    shuffle_str = "" if args.shuffle_idx is None else f"_shuffle_{args.shuffle_idx}"
    return f"{args.feat}_{args.condition}{shuffle_str}"

def get_selected_features_cross_cond_file_name(args):
    """
    Naming convention for preferred beliefs decoding files
    """
    return f"{args.feat}_cross_{args.model_cond}_model_on_{args.data_cond}_data"

def get_selected_features_cross_time_file_name(args):
    """
    Naming convention for preferred beliefs decoding files
    """
    return f"{args.feat}_{args.condition}_cross_time"

def get_selected_features_output_dir(args, make_dir=True):
    """
    Directory convention for preferred beliefs decoding
    """
    region_str = "" if args.regions is None else f"_{args.regions.replace(',', '_').replace(' ', '_')}"
    fr_type_str = f"_{args.fr_type}" if args.fr_type != "firing_rates" else ""
    filt_str = "".join([f"_{k}_{v}"for k, v in args.beh_filters.items()])
    v2_pseudo_str = "_v2_pseudo" if args.use_v2_pseudo else ""
    balance_str = "_balanced" if args.balance_by_filters else ""
    sig_units_str = f"_{args.sig_unit_level}_units" if args.sig_unit_level else ""
    
    run_name = f"{args.subject}_{args.trial_event}{fr_type_str}{region_str}{filt_str}{v2_pseudo_str}{balance_str}{sig_units_str}"
    if args.shuffle_idx is None: 
        dir = os.path.join(args.base_output_path, f"{run_name}")
    else: 
        dir = os.path.join(args.base_output_path, f"{run_name}/{args.shuffle_method}_shuffles")
    if make_dir: 
        os.makedirs(dir, exist_ok=True)
    return dir

def get_anova_split_path(args):
    filt_str = "_".join([f"{k}_{v}"for k, v in args.beh_filters.items()])
    return f"/data/patrick_res/sessions/{args.subject}/belief_partition_splits_{filt_str}.pickle"

def get_anova_file_name(args):
    shuffle_str = "" if args.shuffle_idx is None else f"_shuffle_{args.shuffle_idx}"
    return f"{args.feat}_{shuffle_str}"
    
def get_anova_output_dir(args, make_dir=True):
    condition_str = "_".join(args.conditions)
    time_range_str = f"{args.time_range[0]}_to_{args.time_range[1]}" if args.time_range else None
    filt_str = "_".join([f"{k}_{v}"for k, v in args.beh_filters.items()])
    window_str = f"window_{args.window_size}" if args.window_size else None
    split_str = f"split_{args.split_idx}" if args.split_idx is not None else None
    components = [args.subject, args.trial_event, condition_str, time_range_str, filt_str, window_str, split_str]
    run_name = "_".join(s for s in components if s)
    if args.shuffle_idx is None: 
        dir = os.path.join(args.base_output_path, f"{run_name}")
    else: 
        dir = os.path.join(args.base_output_path, f"{run_name}/{args.shuffle_method}_shuffles")
    if make_dir: 
        os.makedirs(dir, exist_ok=True)
    return dir    

def transform_np_acc_to_df(acc, args):
    """
    Takes accuracy np array of n_time x n_runs, 
    Converts to dataframe of columns Time, run, Accuracy
    """
    df = pd.DataFrame(acc).reset_index(names=["Time"])
    ti = args.trial_interval
    df["Time"] = (df["Time"] * ti.interval_size + ti.interval_size - ti.pre_interval) / 1000
    df = df.melt(id_vars="Time", value_vars=list(range(acc.shape[1])), var_name="run", value_name="Accuracy")
    return df


def load_ccgp_value_df_from_pairs(args, pairs, dir, conds, shuffle=False):
    res = []
    for i, row in pairs.iterrows():
        # NOTE: hack, need to run 18 runs instead of 17.
        if i < 17:
            args.feat_pair = row.pair
            file_name = get_ccgp_val_file_name(args)
            for cond in conds: 
                try: 
                    full_path = os.path.join(dir, f"{file_name}_{cond}_accs.npy")
                    acc = np.load(full_path)
                except Exception as e:
                    if shuffle:
                        print(f"Warning, shuffle not found: {full_path}")
                        continue
                    else: 
                        raise e
                df = transform_np_acc_to_df(acc, args)
                # df["pair"] = feat1, feat2
                df["condition"] = cond if not shuffle else f"{cond}_shuffle"
                res.append(df)
    return pd.concat(res)

def read_ccgp_value(args, pairs, conds=["within_cond", "across_cond", "overall"], num_shuffles=10):
    """
    Returns two dataframes, one for ccgp one for shuffles
    """
    args.trial_interval = get_trial_interval(args.trial_event)
    args.shuffle_idx = None
    dir = get_ccgp_val_output_dir(args, make_dir=False)
    res = load_ccgp_value_df_from_pairs(args, pairs, dir, conds)
    shuffle_res = []
    for shuffle_idx in range(num_shuffles):
        args.shuffle_idx = shuffle_idx
        dir = get_ccgp_val_output_dir(args, make_dir=False)
        shuffle_res.append(load_ccgp_value_df_from_pairs(args, pairs, dir, conds, shuffle=True))
    res = pd.concat(([res] + shuffle_res))
    return res

def get_ccgp_feat_model_for_pair(args, dir, feat, feat_pair):
    args.feat_pair = feat_pair
    file_name = get_ccgp_val_file_name(args)
    models = np.load(os.path.join(dir, f"{file_name}_feat_{feat}_models.npy"), allow_pickle=True)
    df = pd.DataFrame(models).reset_index(names=["Time"])
    ti = args.trial_interval
    df["Time"] = (df["Time"] * ti.interval_size + ti.interval_size - ti.pre_interval) / 1000
    df = df.melt(id_vars="Time", value_vars=list(range(models.shape[1])), var_name="run", value_name="models")
    # set pair column as pair arr for every row
    df["pair"] = [feat_pair] * len(df)
    df["feat"] = feat
    return df


def read_ccgp_models(args, pairs):
    """
    Returns df with all the models for single selected feature decoding
    """
    args.trial_interval = get_trial_interval(args.trial_event)
    dir = get_ccgp_val_output_dir(args, make_dir=False)
    res = []
    for i, row in pairs.iterrows():
        pair = row.pair
        res.append(get_ccgp_feat_model_for_pair(args, dir, pair[0], pair))
        res.append(get_ccgp_feat_model_for_pair(args, dir, pair[1], pair))
    return pd.concat(res)

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

def load_preferred_belief_df_from_pairs(args, pairs, dir, shuffle=False):
    res = []
    for i, row in pairs.iterrows():
        for chosen_not_pref in [True, False]: 
            args.feat_pair = row.pair
            args.chosen_not_preferred = chosen_not_pref
            file_name = get_preferred_beliefs_file_name(args)
            full_path = os.path.join(dir, f"{file_name}_test_accs.npy")
            try: 
                acc = np.load(full_path)
            except Exception as e:
                if shuffle:
                    print(f"Warning, shuffle not found: {full_path}")
                    continue
                else: 
                    raise e

            df = transform_np_acc_to_df(acc, args)
            pref_str = "not_pref" if chosen_not_pref else "pref"
            shuffle_str = "_shuffle" if shuffle else ""
            df["condition"] = f"{pref_str}{shuffle_str}"
            res.append(df)
    return pd.concat(res)

def read_preferred_beliefs(args, pairs, num_shuffles=10):
    """
    Returns two dataframes, one for ccgp one for shuffles
    """
    args.shuffle_idx = None
    args.trial_interval = get_trial_interval(args.trial_event)
    dir = get_preferred_beliefs_output_dir(args, make_dir=False)
    res = load_preferred_belief_df_from_pairs(args, pairs, dir)
    shuffle_res = []
    for shuffle_idx in range(num_shuffles):
        args.shuffle_idx = shuffle_idx
        dir = get_preferred_beliefs_output_dir(args, make_dir=False)
        shuffle_res.append(load_preferred_belief_df_from_pairs(args, pairs, dir, shuffle=True))
    res = pd.concat(([res] + shuffle_res))
    return res

def load_selected_features_df(args, feats, dir, conds, shuffle=False):
    res = []
    for feat in feats:
        for condition in conds: 
            args.feat = feat
            args.condition = condition
            file_name = get_selected_features_file_name(args)
            try: 
                full_path = os.path.join(dir, f"{file_name}_test_accs.npy")
                acc = np.load(full_path)
            except Exception as e:
                if shuffle:
                    print(f"Warning, shuffle not found: {file_name}")
                    continue
                else: 
                    raise e
            df = transform_np_acc_to_df(acc, args)
            shuffle_str = "_shuffle" if shuffle else ""
            df["condition"] = f"{condition}{shuffle_str}"
            df["feat"] = feat
            res.append(df)
    return pd.concat(res)

def read_selected_features(args, feats, conds=["chosen", "pref", "not_pref"], num_shuffles=10):
    """
    Returns two dataframes, one for ccgp one for shuffles
    """
    args.shuffle_idx = None
    args.trial_interval = get_trial_interval(args.trial_event)
    dir = get_selected_features_output_dir(args, make_dir=False)
    res = load_selected_features_df(args, feats, dir, conds)
    shuffle_res = []
    for shuffle_idx in range(num_shuffles):
        args.shuffle_idx = shuffle_idx
        dir = get_selected_features_output_dir(args, make_dir=False)
        shuffle_res.append(load_selected_features_df(args, feats, dir, conds, shuffle=True))
    res = pd.concat(([res] + shuffle_res))
    return res  

def read_selected_features_models(args, feats, cond):
    """
    Returns df with all the models for single selected feature decoding
    """
    args.trial_interval = get_trial_interval(args.trial_event)
    dir = get_selected_features_output_dir(args, make_dir=False)
    res = []
    for feat in feats:
        args.feat = feat
        args.condition = cond
        file_name = get_selected_features_file_name(args)
        models = np.load(os.path.join(dir, f"{file_name}_models.npy"), allow_pickle=True)
        df = pd.DataFrame(models).reset_index(names=["Time"])
        ti = args.trial_interval
        df["Time"] = (df["Time"] * ti.interval_size + ti.interval_size - ti.pre_interval) / 1000
        df = df.melt(id_vars="Time", value_vars=list(range(models.shape[1])), var_name="run", value_name="models")
        df["feat"] = feat
        res.append(df)
    return pd.concat(res)


def get_selected_features_weights(models):
    models["weights"] = models.apply(lambda x: x.models.coef_[0, :], axis=1)
    return models[["Time", "feat", "run", "weights"]]

def get_weights_per_model(row, unit_ids):
    return pd.DataFrame({
        "PseudoUnitID": unit_ids.PseudoUnitID,
        "weight": row.models.coef_[0, :],
        "Time": row.Time,
        "run": row.run,
    })

def get_per_feat_unit_weights(group, args):
    args.feat = group.name
    dir = get_selected_features_output_dir(args, make_dir=False)
    file_name = get_selected_features_file_name(args)
    units = pd.read_csv(os.path.join(dir, f"{file_name}_unit_ids.csv"), names=["idx", "PseudoUnitID"], skiprows=1)
    # pseudo unit ID indexes in models go by sorted PseudoUnitIDs
    sorted = units.sort_values(by="PseudoUnitID")
    df = pd.concat(group.apply(lambda row: get_weights_per_model(row, sorted), axis=1).values)
    df["feat"] = group.name
    return df
    

def get_selected_features_weights_with_ids(args, feats, cond):
    """
    Want to return df with columns: feat, run, time, pseudo_unit_id, weight
    """
    args.cond = cond
    models = read_selected_features_models(args, feats, cond)
    return models.groupby("feat").apply(lambda x: get_per_feat_unit_weights(x, args)).reset_index(drop=True)


def read_selected_features_cross_cond(args, feats, cond_pair): 
    res = []
    dir = get_selected_features_output_dir(args, make_dir=False)
    args.trial_interval = get_trial_interval(args.trial_event)
    for feat in feats: 
        for i in range(2):
            args.feat = feat
            args.model_cond = cond_pair[i]
            args.data_cond = cond_pair[(i + 1) % 2]
            file_name = get_selected_features_cross_cond_file_name(args)
            accs = np.load(os.path.join(dir, f"{file_name}_accs.npy"))
            df = transform_np_acc_to_df(accs, args)
            df["condition"] = f"{args.model_cond} model on {args.data_cond} data"
            res.append(df)
    return pd.concat(res)

def read_selected_features_cross_time(args, feats, cond, avg=False):
    df = []
    dir = get_selected_features_output_dir(args, make_dir=False)
    args.trial_interval = get_trial_interval(args.trial_event)
    args.condition = cond
    for feat in feats:
        args.feat = feat
        file_name = get_selected_features_cross_time_file_name(args)
        accs = np.load(os.path.join(dir, f"{file_name}_accs.npy"))
        ti = args.trial_interval
        for (train_time_idx, test_time_idx, run_idx), acc in np.ndenumerate(accs):
            train_time =  (train_time_idx * ti.interval_size + ti.interval_size - ti.pre_interval) / 1000
            test_time =  (test_time_idx * ti.interval_size + ti.interval_size - ti.pre_interval) / 1000
            df.append({"TrainTime": train_time, "TestTime": test_time, "RunIdx": run_idx, "Feat": feat, "Accuracy": acc})
    df = pd.DataFrame(df)
    if avg: 
        return df.groupby(["TrainTime", "TestTime"]).Accuracy.mean().reset_index(name="Accuracy")
    else: 
        return df
    

def read_anova_good_units(args, percentile_str="95th", cond="combined_fracvar", return_pos=True, read_shuffle=True):
    args.trial_interval = get_trial_interval(args.trial_event)
    output_dir = get_anova_output_dir(args, make_dir=False)
    good_res = []
    for feat in FEATURES:
        res = pd.read_pickle(os.path.join(output_dir, f"{feat}_.pickle"))
        if read_shuffle:
            shuffle_stats = pd.read_pickle(os.path.join(output_dir, f"{feat}_shuffle_stats.pickle"))
            if args.window_size is None: 
                res = pd.merge(res, shuffle_stats, on="PseudoUnitID")
            else: 
                res = pd.merge(res, shuffle_stats, on=["PseudoUnitID", "WindowStartMilli"], how="outer")
        # HACK: this is for backwards compatability, previously only interested in one col
        # named "combined" at a time. 
        if cond != "combined_fracvar":
            cond_col = f"x_{cond}_comb_time_fracvar"
            percentile_col = f"{cond}_{percentile_str}"
        else: 
            cond_col = cond
            percentile_col = percentile_str
        if percentile_str != "all":
            good_res.append(res[res[cond_col] > res[percentile_col]])
        else:
            good_res.append(res)
    good_res = pd.concat(good_res)
    if return_pos:
        unit_pos = pd.read_pickle(UNITS_PATH.format(sub=args.subject))
        good_res = pd.merge(good_res, unit_pos[["PseudoUnitID", "drive", "structure_level2", "structure_level2_cleaned"]])
        good_res["whole_pop"] = "all_regions"
    good_res["trial_event"] = args.trial_event
    return good_res


def read_anova_res_all_time(args, percentile_str="95th", cond="combined_fracvar", return_pos=True, read_shuffle=True):
    args.trial_event = "StimOnset"
    stim_res = read_anova_good_units(args, percentile_str, cond, return_pos, read_shuffle)
    stim_res["abs_time"] = stim_res.WindowEndMilli

    args.trial_event = "FeedbackOnsetLong"
    fb_res = read_anova_good_units(args, percentile_str, cond, return_pos, read_shuffle)
    fb_res["abs_time"] = fb_res.WindowEndMilli + 2400

    all_res = pd.concat((stim_res, fb_res)).reset_index()

    return stim_res, fb_res, all_res


def get_frs_from_args(args, sess_name):
    """
    TODO: deprecate, move to use spike_utils.get_frs_from_args
    """
    trial_interval = args.trial_interval
    spikes_path = SESS_SPIKES_PATH.format(
        sub=args.subject,
        sess_name=sess_name, 
        fr_type=args.fr_type,
        pre_interval=trial_interval.pre_interval, 
        event=trial_interval.event, 
        post_interval=trial_interval.post_interval, 
        interval_size=trial_interval.interval_size
    )
    frs = pd.read_pickle(spikes_path)
    frs["PseudoUnitID"] = int(sess_name) * 100 + frs.UnitID.astype(int)
    # create a time field as well that's relative to the trial event
    frs["Time"] = frs["TimeBins"] - args.trial_interval.pre_interval / 1000
    if hasattr(args, "time_range") and args.time_range is not None: 
        if len(args.time_range) !=2: 
            raise ValueError("must have two ranges")
        # time_range specified in milliseconds, relative to trial event, convert to 
        # be in seconds, relative to pre_interval
        start, end = [(x + args.trial_interval.pre_interval) / 1000 for x in args.time_range]
        frs = frs[(frs.TimeBins >= start) & (frs.TimeBins < end)]
    return frs


