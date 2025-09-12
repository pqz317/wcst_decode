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
    if "feat_pair" in args: 
        return get_ccgp_file_name(args)
    else: 
        shuffle_str = "" if args.shuffle_idx is None else f"_shuffle_{args.shuffle_idx}"
        return f"{args.feat}_{args.mode}{shuffle_str}"

def get_cross_time_file_name(args):
    """
    Naming convention for preferred beliefs decoding files
    """
    model_event_str = f"_{args.model_trial_event}_model" if args.model_trial_event else ""
    return f"{args.feat}_{args.mode}_cross_time{model_event_str}"

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
                # print(f"Warning, shuffle not found: {file_name}")
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
    args.shuffle_idx = None
    res = pd.concat(([res] + shuffle_res))
    return res  

def read_units(args, feats):
    args.trial_interval = get_trial_interval(args.trial_event)
    dir = get_dir_name(args, make_dir=False)
    res = []
    for feat in feats:
        args.feat = feat
        file_name = get_file_name(args)
        if "feat_pair" in args: 
            df = pd.read_csv(os.path.join(dir, f"{file_name}_feat_{feat}_unit_ids.csv"))
        else: 
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
            df["dim_type"] = row.dim_type
            df["pair_str"] = "_".join(row.pair)
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
    res["TimeIdx"] = (res.Time * 10).astype(int)
    return res

def get_ccgp_feat_model_for_pair(args, dir, feat, feat_pair):
    args.trial_interval = get_trial_interval(args.trial_event)
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

def get_contributions_for_all_time(args, region_level, sig_region_thresh=20, run_idx=None, feat_agg_func="mean", events=["StimOnset", "FeedbackOnsetLong"]):
    all_conts = []
    for event in events: 
        args.trial_event = event 
        conts = read_contributions(args, region_level, sig_region_thresh, run_idx, feat_agg_func)
        conts["abs_time"] = conts.Time + 2.8 if event == "FeedbackOnsetLong" else conts.Time
        all_conts.append(conts)
    all_conts = pd.concat(all_conts).reset_index()
    return all_conts

def get_weights(args):
    high_idx = MODE_TO_CLASSES[args.mode].index(MODE_TO_DIRECTION_LABELS[args.mode]["high"])
    low_idx = MODE_TO_CLASSES[args.mode].index(MODE_TO_DIRECTION_LABELS[args.mode]["low"])

    if "feat_pair" in args:
        models = get_ccgp_feat_model_for_pair(args, get_dir_name(args, make_dir=False), args.feat, args.feat_pair)
    else: 
        models = read_models(args, [args.feat])
    unit_ids = read_units(args, [args.feat])

    models["weightsdiff"] = models.apply(lambda x: x.models.coef_[high_idx, :] - x.models.coef_[low_idx, :], axis=1)
    models["batch_mean"] = models.apply(lambda x: x.models.model.norm.running_mean.detach().cpu().numpy(), axis=1)
    # 1e-5 from torch batchnorm1d, numerical 
    models["batch_std"] = models.apply(lambda x: np.sqrt(x.models.model.norm.running_var.detach().cpu().numpy() + 1e-5), axis=1)
    models["TimeIdx"] = (models["Time"] * 10).astype(int)

    def avg_and_label(x):
        weights_diff_means = np.mean(np.vstack(x.weightsdiff.values), axis=0)
        mean_means = np.mean(np.vstack(x.batch_mean.values), axis=0)
        std_means = np.mean(np.vstack(x.batch_std.values), axis=0)
        weights_diff_normed = weights_diff_means / std_means
        pos = np.arange(len(weights_diff_means))
        
        return pd.DataFrame({"pos": pos, "weightsdiff": weights_diff_means, "weightsdiff_normed": weights_diff_normed, "mean": mean_means, "std": std_means})
    weights = models.groupby(["TimeIdx", "feat"]).apply(avg_and_label).reset_index()
    weights = pd.merge(weights, unit_ids, on=["feat", "pos"])
    return weights

def load_update_df(args, feats, dir, shuffle=False):
    res = []
    for feat in feats:
        args.feat = feat
        file_name = get_file_name(args)
        try: 
            full_path = os.path.join(dir, f"{file_name}_projections.pickle")
            proj = pd.read_pickle(full_path)
        except Exception as e:
            if shuffle:
                print(f"Warning, shuffle not found: {file_name}")
                continue
            else: 
                raise e
        shuffle_str = "_shuffle" if shuffle else ""
        proj["mode"] = f"{args.mode}{shuffle_str}"
        proj["feat"] = feat
        res.append(proj)
    return pd.concat(res)

def read_update_projections(args, num_shuffles=3):
    args.shuffle_idx = None
    args.trial_interval = get_trial_interval(args.trial_event)
    args.base_output_path = "/data/patrick_res/update_projections"
    args.beh_filters = args.conditions 
    dir = get_dir_name(args, make_dir=False)
    res = load_update_df(args, FEATURES, dir)
    shuffle_res = []
    for shuffle_idx in range(num_shuffles):
        args.shuffle_idx = shuffle_idx
        dir = get_dir_name(args, make_dir=False)
        shuffle_res.append(load_update_df(args, FEATURES, dir, shuffle=True))
    args.shuffle_idx = None
    res = pd.concat(([res] + shuffle_res))
    return res

def read_update_projections_pvals(args, cond_map, axis_vars=["pref", "conf"]):
    args.base_output_path = "/data/patrick_res/update_projections"
    res = []
    for cond in cond_map:
        for axis_var in axis_vars: 
            args.beh_filters = cond_map[cond] 
            args.mode = axis_var
            args.sig_unit_level = f"{args.mode}_99th_no_cond_window_filter_drift"
            dir = get_dir_name(args, make_dir=False)
            file_name = os.path.join(dir, f"{axis_var}_p_val.txt")
            p = None
            with open(file_name, 'r') as f:
                p = float(f.readline())
            res.append({
                "cond": cond,
                "var": axis_var,
                "p": p
            })
    return pd.DataFrame(res)


def read_similarities(args, pairs):
    all_res = []
    for i, pair in pairs.iterrows():
        args.feat_pair = pair.pair
        out_dir = get_dir_name(args, make_dir=False)
        file_name = get_ccgp_file_name(args)
        low_str = "_to_low" if args.relative_to_low else ""
        res = pd.read_pickle(os.path.join(out_dir, f"{file_name}_{args.sim_type}{low_str}.pickle"))
        res["dim_type"] = pair.dim_type
        res["pair_str"] = "_".join(pair.pair)
        all_res.append(res)
    all_res = pd.concat(all_res)
    all_res["Time"] = all_res["TimeIdx"] / 10
    return all_res

def read_all_similarities(args, pairs, num_shuffles=10):
    args = copy.deepcopy(args)
    true_res = read_similarities(args, pairs)
    true_res["type"] = "true"

    all_shuffles = []
    for i in range(num_shuffles):
        args.shuffle_idx = i
        shuf_res = read_similarities(args, pairs)
        shuf_res["PseudoTrialNumber"] = shuf_res.PseudoTrialNumber * num_shuffles + i
        all_shuffles.append(shuf_res)
    all_shuffles = pd.concat(all_shuffles)
    all_shuffles["type"] = "shuffle"
    all_res = pd.concat((true_res, all_shuffles))
    # still actually dont know why I need this, but plotting for ACC doesn't work without...
    all_res = all_res.reset_index(drop=True)
    return all_res