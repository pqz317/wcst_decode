import os
import scipy.io
import numpy as np
import pandas as pd
import pickle
import torch
import itertools
from constants.glm_constants import *
import copy
from constants.behavioral_constants import *
from constants.decoding_constants import *

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

def get_file_name(args):
    shuffle_str = "" if args.shuffle_idx is None else f"_shuffle_{args.shuffle_idx}"
    return f"{args.feat}_{args.mode}{shuffle_str}"

def get_cross_time_file_name(args):
    """
    Naming convention for preferred beliefs decoding files
    """
    model_event_str = f"_{args.model_trial_event}_model" if args.model_trial_event else ""
    return f"{args.feat}_{args.mode}_cross_time{model_event_str}"

def get_choice_reward_file_name(args):
    return f"{args.feat}_{args.mode}_choice_reward_separate"

def get_ccgp_file_name(args):
    """
    Naming convention for preferred beliefs decoding files
    """
    pair_str = pair_str = "_".join(args.feat_pair)
    shuffle_str = "" if args.shuffle_idx is None else f"_shuffle_{args.shuffle_idx}"
    return f"{pair_str}_{args.mode}{shuffle_str}"

def get_dir_name(args, make_dir=True):
    """
    Directory convention for preferred beliefs decoding
    """
    region_str = "" if args.regions is None else f"{args.regions.replace(',', '_').replace(' ', '_')}"
    filt_str = "_".join([f"{k}_{v}"for k, v in args.beh_filters.items()])
    sig_units_str = f"{args.sig_unit_level}_units" if args.sig_unit_level else None
    splitter_str = f"kfold_{args.num_splits}" if args.splitter == "kfold" else None
    parts = [args.subject, args.trial_event, region_str, filt_str, sig_units_str, splitter_str]
    run_name = "_".join(x for x in parts if x)
    if args.shuffle_idx is None: 
        dir = os.path.join(args.base_output_path, f"{run_name}")
    else: 
        dir = os.path.join(args.base_output_path, f"{run_name}/shuffles")
    if make_dir: 
        os.makedirs(dir, exist_ok=True)
    return dir

def load_df(args, feats, dir, shuffle=False):
    res = []
    for feat in feats:
        args.feat = feat
        file_name = get_file_name(args)
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
        df["mode"] = f"{args.mode}{shuffle_str}"
        df["feat"] = feat
        res.append(df)
    return pd.concat(res)


def read_results(args, feats, num_shuffles=10):
    """
    Returns two dataframes, one for ccgp one for shuffles
    """
    args.shuffle_idx = None
    args.trial_interval = get_trial_interval(args.trial_event)
    dir = get_dir_name(args, make_dir=False)
    res = load_df(args, feats, dir)
    shuffle_res = []
    for shuffle_idx in range(num_shuffles):
        args.shuffle_idx = shuffle_idx
        dir = get_dir_name(args, make_dir=False)
        shuffle_res.append(load_df(args, feats, dir, shuffle=True))
    res = pd.concat(([res] + shuffle_res))
    return res  

def read_units(args, feats):
    args.trial_interval = get_trial_interval(args.trial_event)
    dir = get_dir_name(args, make_dir=False)
    res = []
    for feat in feats:
        args.feat = feat
        file_name = get_file_name(args)
        df = pd.read_csv(os.path.join(dir, f"{file_name}_unit_ids.csv"))
        # fixing a bug here...
        df = df.rename(columns={"PseudoUnitIDs": "PseudoUnitID"})
        df = df.sort_values(by="PseudoUnitID")
        df["pos"] = range(len(df))
        df["feat"] = feat
        res.append(df)
    return pd.concat(res)

def read_models(args, feats):
    """
    Returns df with all the models for single selected feature decoding
    """
    args.trial_interval = get_trial_interval(args.trial_event)
    dir = get_dir_name(args, make_dir=False)
    res = []
    for feat in feats:
        args.feat = feat
        file_name = get_file_name(args)
        models = np.load(os.path.join(dir, f"{file_name}_models.npy"), allow_pickle=True)
        df = pd.DataFrame(models).reset_index(names=["Time"])
        ti = args.trial_interval
        df["Time"] = (df["Time"] * ti.interval_size + ti.interval_size - ti.pre_interval) / 1000
        df = df.melt(id_vars="Time", value_vars=list(range(models.shape[1])), var_name="run", value_name="models")
        df["feat"] = feat
        res.append(df)
    return pd.concat(res)

def read_cross_time_results(args, feats, avg=False):
    df = []
    args.shuffle_idx = None
    dir = get_dir_name(args, make_dir=False)
    args.trial_interval = get_trial_interval(args.trial_event)
    for feat in feats:
        args.feat = feat
        file_name = get_cross_time_file_name(args)
        accs = np.load(os.path.join(dir, f"{file_name}_accs.npy"))
        model_ti = get_trial_interval(args.model_trial_event) if args.model_trial_event else get_trial_interval(args.trial_event)
        ti =  get_trial_interval(args.trial_event)
        for (train_time_idx, test_time_idx, run_idx), acc in np.ndenumerate(accs):
            train_time =  (train_time_idx * model_ti.interval_size + model_ti.interval_size - model_ti.pre_interval) / 1000
            test_time =  (test_time_idx * ti.interval_size + ti.interval_size - ti.pre_interval) / 1000
            df.append({"TrainTime": train_time, "TestTime": test_time, "RunIdx": run_idx, "Feat": feat, "Accuracy": acc})
    df = pd.DataFrame(df)
    if avg: 
        df = df.groupby(["TrainTime", "TestTime"]).Accuracy.mean().reset_index(name="Accuracy")
    df["TrainEvent"] = args.model_trial_event if args.model_trial_event else args.trial_event
    df["TestEvent"] = args.trial_event
    return df
    
def load_ccgp_df(args, pairs, dir, conds, shuffle=False):
    res = []
    for i, row in pairs.iterrows():
        # NOTE: hack, need to run 18 runs instead of 17.
        if i < 17:
            args.feat_pair = row.pair
            file_name = get_ccgp_file_name(args)
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

def read_ccgp_results(args, pairs, conds=["within_cond", "across_cond", "overall"], num_shuffles=10):
    """
    Returns two dataframes, one for ccgp one for shuffles
    """
    args.trial_interval = get_trial_interval(args.trial_event)
    args.shuffle_idx = None
    dir = get_dir_name(args, make_dir=False)
    res = load_ccgp_df(args, pairs, dir, conds)
    shuffle_res = []
    for shuffle_idx in range(num_shuffles):
        args.shuffle_idx = shuffle_idx
        dir = get_dir_name(args, make_dir=False)
        shuffle_res.append(load_ccgp_df(args, pairs, dir, conds, shuffle=True))
    args.shuffle_idx = None
    res = pd.concat(([res] + shuffle_res))
    return res

def get_ccgp_feat_model_for_pair(args, dir, feat, feat_pair):
    args.feat_pair = feat_pair
    file_name = get_ccgp_file_name(args)
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
    dir = get_dir_name(args, make_dir=False)
    res = []
    for i, row in pairs.iterrows():
        pair = row.pair
        res.append(get_ccgp_feat_model_for_pair(args, dir, pair[0], pair))
        res.append(get_ccgp_feat_model_for_pair(args, dir, pair[1], pair))
    return pd.concat(res)

def read_contributions(args, region_level="whole_pop", sig_region_thresh=20, run_idx=None, feat_agg_func="mean"):
    models = read_models(args, FEATURES)
    unit_ids = read_units(args, FEATURES)
    if run_idx is not None: 
        models = models[models.run == run_idx]
    models["weightsdiffabs"] = models.apply(lambda x: np.abs(x.models.coef_[0, :] - x.models.coef_[1, :]), axis=1)
    def avg_and_label(x):
        means = np.mean(np.vstack(x.weightsdiffabs.values), axis=0)
        pos = np.arange(len(means))
        return pd.DataFrame({"pos": pos, "contribution": means})
    conts = models.groupby(["Time", "feat"]).apply(avg_and_label).reset_index()
    conts = pd.merge(conts, unit_ids, on=["feat", "pos"])

    col_name = f"{feat_agg_func}_cont"
    if feat_agg_func == "mean":
        grouped_conts = conts.groupby(["PseudoUnitID", "Time"]).contribution.mean().reset_index(name=col_name)
    elif feat_agg_func == "max": 
        grouped_conts = conts.groupby(["PseudoUnitID", "Time"]).contribution.max().reset_index(name=col_name)

    # assign regions
    if args.subject == "both":
        sa_pos = pd.read_pickle(UNITS_PATH.format(sub="SA"))
        bl_pos = pd.read_pickle(UNITS_PATH.format(sub="BL"))
        units_pos = pd.concat((sa_pos, bl_pos))
    else:
        units_pos = pd.read_pickle(UNITS_PATH.format(sub=args.subject))
    grouped_conts = pd.merge(grouped_conts, units_pos, on="PseudoUnitID")
    grouped_conts["whole_pop"] = "all_regions"

    num_units_by_region = grouped_conts.groupby(region_level).PseudoUnitID.nunique().reset_index(name="num_units")
    sig_regions = num_units_by_region[num_units_by_region.num_units > sig_region_thresh][region_level]
    grouped_conts = grouped_conts[grouped_conts[region_level].isin(sig_regions)]

    grouped_conts["trial_event"] = args.trial_event
    return grouped_conts

def get_contributions_for_all_time(args, region_level, sig_region_thresh=20, run_idx=None, feat_agg_func="mean"):
    args.trial_event = "StimOnset"
    stim_conts = read_contributions(args, region_level, sig_region_thresh, run_idx, feat_agg_func)
    stim_conts["abs_time"] = stim_conts.Time

    args.trial_event = "FeedbackOnsetLong"
    fb_conts = read_contributions(args, region_level, sig_region_thresh, run_idx, feat_agg_func)
    fb_conts["abs_time"] = fb_conts.Time + 2.8

    all_conts = pd.concat((stim_conts, fb_conts)).reset_index()

    return stim_conts, fb_conts, all_conts