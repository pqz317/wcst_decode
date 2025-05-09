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
    return f"{args.feat}_{args.mode}_cross_time"

def get_dir_name(args, make_dir=True):
    """
    Directory convention for preferred beliefs decoding
    """
    region_str = "" if args.regions is None else f"{args.regions.replace(',', '_').replace(' ', '_')}"
    filt_str = "_".join([f"{k}_{v}"for k, v in args.beh_filters.items()])
    sig_units_str = f"{args.sig_unit_level}_units" if args.sig_unit_level else None
    parts = [args.subject, args.trial_event, region_str, filt_str, sig_units_str]
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