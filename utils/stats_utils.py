"""
Utilities for performing statistical tests, specifically permutation tests comparing to some shuffle distribution
"""
from functools import reduce
import operator
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools

def diff_per_group(group, val_col, shuffle_label_col):
    true_mean = group[group[shuffle_label_col] == "true"][val_col].mean()
    shuffle_mean = group[group[shuffle_label_col] == "shuffle"][val_col].mean()
    return true_mean - shuffle_mean

# re-write this to do shuffles per-group
def compute_p_per_group(data, val_col, shuffle_label_col, num_permutes=1000, seed=42):
    true_diff = diff_per_group(data, val_col, shuffle_label_col)
    all_shuffles = []
    rng = np.random.default_rng(seed=seed)
    for i in tqdm(np.arange(num_permutes)):
        shuffle_data = data.copy()
        shuffle_data[shuffle_label_col] = rng.permutation(data[shuffle_label_col].values)
        shuffle_diff = diff_per_group(shuffle_data, val_col, shuffle_label_col)
        all_shuffles.append(shuffle_diff)
    return np.mean(true_diff <= np.array(all_shuffles))

def get_n_time_offset(args):
    if args.trial_event == "StimOnset":
        n_time = 20
        offset = 0.9
    else: 
        n_time = 33
        offset = 1.7
    return n_time, offset

def compute_p_for_decoding_by_time(res, args): 
    res["shuffle_type"] = res["mode"].map({"pref": "true", "pref_shuffle": "shuffle"})
    n_time, offset = get_n_time_offset(args)
    p_res = []
    for time_idx in tqdm(range(n_time)):
        time = round(time_idx / 10 - offset, 1)
        time_res = res[np.isclose(res.Time, time)]
        p = compute_p_per_group(time_res, "Accuracy", "shuffle_type")
        p_res.append({"Time": time, "TimeIdx": time_idx, "p": p})
    p_res = pd.DataFrame(p_res)
    return p_res

def compute_p_for_cross_decoding_by_time(cross_res, shuffles, args): 
    n_time, offset = get_n_time_offset(args)
    p_res = []
    for (train_idx, test_idx) in tqdm(itertools.product(range(n_time), range(n_time))):
        train_time = round(train_idx / 10 - offset, 1)
        test_time = round(test_idx / 10 - offset, 1)

        time_res = cross_res[np.isclose(cross_res.TrainTime, train_time) & np.isclose(cross_res.TestTime, test_time)].copy()
        time_res["shuffle_type"] = "true"
        shuffle_time_res = shuffles[np.isclose(shuffles.Time, test_time)].copy()
        shuffle_time_res["shuffle_type"] = "shuffle"

        p = compute_p_per_group(pd.concat((time_res, shuffle_time_res)), "Accuracy", "shuffle_type")
        p_res.append({"TrainTime": train_time, "TestTime": test_time, "TrainIdx": train_idx, "TestIdx": test_idx, "p": p})
    p_res = pd.DataFrame(p_res)
    return p_res