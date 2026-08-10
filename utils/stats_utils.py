"""
Utilities for performing statistical tests, specifically permutation tests comparing to some shuffle distribution
"""
from functools import reduce
import operator
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
import copy
from numba import njit

@njit
def perm_test(values, mask_a, num_permutes, rng, one_sided=True):

    # Boolean mask for observed groups
    true_diff = values[mask_a].mean() - values[~mask_a].mean()

    # Permutation differences
    diffs = np.empty(num_permutes)
    for i in range(num_permutes):
        permuted_a = rng.permutation(mask_a)
        diffs[i] = values[permuted_a].mean() - values[~permuted_a].mean()

    # Compute p-value
    if one_sided:
        p_val = np.mean(diffs >= true_diff)
    else:
        p_val = np.mean(np.abs(diffs) >= np.abs(true_diff))
    return p_val


def compute_p_per_group(data, val_col, label_col, num_permutes=1000, seed=42, label_a="true", label_b="shuffle", test_type="one_side"):
    """
    Computes a one-sided permutation test, 
    if one-side: provides p value for label_a > label_b 
    If two-side: p(|label_a - label_b| >= |obs|).
    """
    rng = np.random.default_rng(seed)

    # ensure we just have a, b, rows
    sub_data = data[data[label_col].isin([label_a, label_b])]

    values = sub_data[val_col].values.astype(np.float64)
    mask_a = sub_data[label_col].values == label_a

    one_sided = (test_type == "one_side")
    return perm_test(values, mask_a, num_permutes, rng, one_sided)
    
def get_permutation_test_func(test_type="one_side"):
    def permutation_test_wrapper(pair, data1, data2):
        """
        wrapper for permutation test, used for adding significance markers to bar plots
        calls compute_p_per_group under the hood
        """
        df1 = pd.DataFrame({"label": "a", "vals": data1})
        df2 = pd.DataFrame({"label": "b", "vals": data2})
        df = pd.concat((df1, df2))
        p = compute_p_per_group(df, val_col="vals", label_col="label", label_a="a", label_b="b", test_type=test_type)
        return p
    return permutation_test_wrapper

def get_permutation_test_func_single(shuffle_vals, test_type="one_side"):
    def permutation_test_wrapper(cond, data):
        """
        wrapper for permutation test, used for adding significance markers to bar plots
        calls compute_p_per_group under the hood
        """
        df1 = pd.DataFrame({"label": "a", "vals": data})
        df2 = pd.DataFrame({"label": "b", "vals": shuffle_vals})
        df = pd.concat((df1, df2))
        p = compute_p_per_group(df, val_col="vals", label_col="label", label_a="a", label_b="b", test_type=test_type)
        return p
    return permutation_test_wrapper


def get_n_time_offset(trial_event):
    if trial_event == "StimOnset":
        n_time = 20
        offset = 0.9
    else: 
        n_time = 33
        offset = 1.7
    return n_time, offset

def compute_p_for_decoding_by_time(res, args, val_col="Accuracy"):
    """
    val_col is the column holding the statistic being tested, "Accuracy" for decoding runs.
    Pass e.g. "cos_sim" for analyses whose statistic isn't an accuracy.
    """
    # res["shuffle_type"] = res["mode"].map({"pref": "true", "pref_shuffle": "shuffle"})
    res["shuffle_type"] = res["mode"].apply(lambda x: "shuffle" if "shuffle" in x else "true")
    n_time, offset = get_n_time_offset(args.trial_event)
    p_res = []
    for time_idx in tqdm(range(n_time)):
        time = round(time_idx / 10 - offset, 1)
        time_res = res[np.isclose(res.Time, time)]
        p = compute_p_per_group(time_res, val_col, "shuffle_type")
        p_res.append({"Time": time, "TimeIdx": time_idx, "p": p})
    p_res = pd.DataFrame(p_res)
    return p_res


def _gap_by_time_feat(res):
    """
    Given a read_results() dataframe for a single partition (cols: Time, feat, mode, Accuracy),
    returns a df with columns [Time, feat, gap], where
        gap = mean(true accuracy) - mean(shuffle accuracy)
    averaged over runs (true) and over runs x shuffles (shuffle), per (Time, feat).
    """
    res = res.copy()
    res["shuffle_type"] = res["mode"].apply(lambda x: "shuffle" if "shuffle" in x else "true")
    means = res.groupby(["Time", "feat", "shuffle_type"]).Accuracy.mean().reset_index()
    pivoted = means.pivot_table(index=["Time", "feat"], columns="shuffle_type", values="Accuracy")
    # gap is only defined when both true and shuffle are present for the (Time, feat)
    pivoted = pivoted.dropna(subset=["true", "shuffle"])
    pivoted["gap"] = pivoted["true"] - pivoted["shuffle"]
    return pivoted["gap"].reset_index()


def compute_p_for_dod_by_time(res_in, res_notin, args, seed=42):
    """
    Difference-of-differences significance test, per timepoint:
    tests whether the "In X Dim" partition decodes reliably further above its own shuffle
    baseline than the "Not in X Dim" partition does above its own shuffle baseline, i.e.
        delta_f = (true_In - shuffle_In) - (true_NotIn - shuffle_NotIn)   [per feature f]
        Delta   = mean_f delta_f                                          [observed statistic]

    res_in / res_notin are read_results() outputs for the two partitions (cols: Time, run,
    Accuracy, mode, feat; mode in {args.mode, f"{args.mode}_shuffle"}).

    Null model: per-feature sign-flip. Swapping a feature's In/Not-in label moves its true and
    matched shuffle together, so it is exactly delta_f -> -delta_f. All 2^n_feat sign vectors are
    enumerated exactly (n_feat = # features present in BOTH partitions at that time bin), giving an
    exact one-sided p = mean(Delta* >= Delta). Because enumeration includes the identity (all +1),
    p is always >= 1 / 2^n_feat. `seed` is unused (enumeration is deterministic); kept for a
    consistent signature with the sibling permutation functions.

    Returns df[Time, TimeIdx, p, n_feat] (Time/TimeIdx/p schema matches compute_p_for_decoding_by_time).
    """
    gap_in = _gap_by_time_feat(res_in).rename(columns={"gap": "gap_in"})
    gap_notin = _gap_by_time_feat(res_notin).rename(columns={"gap": "gap_notin"})
    # inner merge -> keep only features present in BOTH partitions at a given time bin
    merged = pd.merge(gap_in, gap_notin, on=["Time", "feat"], how="inner")
    merged["delta"] = merged["gap_in"] - merged["gap_notin"]

    n_time, offset = get_n_time_offset(args.trial_event)
    p_res = []
    for time_idx in tqdm(range(n_time)):
        time = round(time_idx / 10 - offset, 1)
        deltas_f = merged[np.isclose(merged.Time, time)]["delta"].values.astype(np.float64)
        n_feat = len(deltas_f)
        if n_feat == 0:
            p = np.nan
        else:
            obs = deltas_f.mean()
            # (2^n_feat, n_feat) matrix of all +/-1 sign vectors
            signs = np.array(list(itertools.product([1.0, -1.0], repeat=n_feat)))
            null_deltas = signs @ deltas_f / n_feat
            p = np.mean(null_deltas >= obs)
        p_res.append({"Time": time, "TimeIdx": time_idx, "p": p, "n_feat": n_feat})
    return pd.DataFrame(p_res)


def compute_p_for_cross_decoding_by_time(cross_res, shuffles, args):
    train_event = args.model_trial_event if args.model_trial_event is not None else args.trial_event
    test_event = args.trial_event

    train_n_time, train_offset = get_n_time_offset(train_event)
    test_n_time, test_offset = get_n_time_offset(test_event)
    
    p_res = []
    for (train_idx, test_idx) in tqdm(itertools.product(range(train_n_time), range(test_n_time))):
        train_time = round(train_idx / 10 - train_offset, 1)
        test_time = round(test_idx / 10 - test_offset, 1)

        time_res = cross_res[np.isclose(cross_res.TrainTime, train_time) & np.isclose(cross_res.TestTime, test_time)].copy()
        time_res["shuffle_type"] = "true"
        shuffle_time_res = shuffles[np.isclose(shuffles.Time, test_time)].copy()
        shuffle_time_res["shuffle_type"] = "shuffle"

        p = compute_p_per_group(pd.concat((time_res, shuffle_time_res)), "Accuracy", "shuffle_type")
        p_res.append({"TrainTime": train_time, "TestTime": test_time, "TrainIdx": train_idx, "TestIdx": test_idx, "p": p})
    p_res = pd.DataFrame(p_res)
    return p_res