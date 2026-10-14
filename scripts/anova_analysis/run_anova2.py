"""
run_anova.py followed by 20250329_aggregate_shuffles.py, in one process per feature, the way
dim_pair_anova.py runs: firing rates are read once per session into a trials x units x bins array,
behavior is labelled once per session and only its TrialNumbers are shuffled, and the decomposition
is computed from cell sums by anova_utils.factorial_fracvar. Same configs (AnovaConfigs), same
results.

A circular shift (or random permutation) moves only TrialNumber, and every label run_anova.load_beh
adds is a function of a row's other columns or of row order, so labelling before the shuffle gives
the same rows. The b split and split_idx subset depend on TrialNumber, so they come after it.

A shuffle in which any session has no trials is dropped for the feature, as run_anova's job for it
would have crashed and the aggregation used the rest.

Writes, per feature, to io_utils.get_anova_output_dir:
    {feat}_.pickle                 the true run, as run_anova
    {feat}_shuffles.pickle         all shuffles, run_anova's columns plus shuffle_idx, in place of
                                   {shuffle_method}_shuffles/{feat}__shuffle_{k}.pickle
    {feat}_shuffle_stats.pickle    as 20250329_aggregate_shuffles.py

    python3 scripts/anova_analysis/run_anova2.py --subject SA --trial_event StimOnset \
        --conditions BeliefConf,BeliefPartition --beh_filters '{"Response":"Correct","Choice":"Chose"}' \
        --window_size 500 [--feat_idx 0] [--num_shuffles 100]
"""

import os
import time
import argparse
import numpy as np
import pandas as pd

import utils.behavioral_utils as behavioral_utils
import utils.io_utils as io_utils
import utils.anova_utils as anova_utils
from constants.behavioral_constants import *
from constants.decoding_constants import *

from scripts.anova_analysis.anova_configs import add_defaults_to_parser, AnovaConfigs
from scripts.pseudo_decoding.belief_partitions.stim_belief_vector_alignment import draw_b_split

FEATS_PATH = "/data/patrick_res/sessions/{sub}/feats_at_least_3blocks.pickle"
SESSIONS_PATH = "/data/patrick_res/sessions/{sub}/valid_sessions.pickle"

SHUFFLE_FUNCS = {
    "circular_shift": lambda beh, k: behavioral_utils.shuffle_beh_by_shift(beh, buffer=50, seed=k),
    "random": lambda beh, k: behavioral_utils.shuffle_beh_random(beh, seed=k),
}


def label_beh(raw, args):
    """
    run_anova.load_beh up to its b split, on unshuffled behavior. The index is raw's row positions
    """
    feat = args.feat
    beh = behavioral_utils.get_belief_partitions(raw.copy(), feat, use_x=args.use_x)
    beh["Choice"] = beh.apply(lambda x: "Chose" if x[FEATURE_TO_DIM[feat]] == feat else "Not Chose", axis=1)
    beh["FeatPreferred"] = beh["PreferredBelief"].apply(lambda x: "Preferred" if x == feat else "Not Preferred")
    return behavioral_utils.filter_behavior(beh, args.beh_filters)


def shuffle_beh(raw, labelled, session, args, shuffle_idx, split_trials):
    """
    The behavior run_anova.load_data would use for shuffle_idx (None for the true run): labelled
    with the TrialNumbers of the shuffled raw behavior, then the b split and split_idx subset
    """
    beh = labelled.copy()
    if shuffle_idx is not None:
        if args.shuffle_method not in SHUFFLE_FUNCS:
            raise ValueError(f"shuffle method {args.shuffle_method} not supported")
        trial_nums = SHUFFLE_FUNCS[args.shuffle_method](raw[["TrialNumber"]].copy(), shuffle_idx).TrialNumber.values
        beh["TrialNumber"] = trial_nums[beh.index]
    if args.b_split_half is not None:
        b = beh[(beh.BeliefPartition == "High Not X") & (beh.Choice == "Chose")].TrialNumber.to_numpy()
        b1, b2 = draw_b_split(session, args.feat, b, args.split_seed, shuffle_idx)
        beh = beh[~beh.TrialNumber.isin(b2 if args.b_split_half == 1 else b1)]
    if split_trials is not None:
        beh = beh[beh.TrialNumber.isin(split_trials)]
    return beh


def load_frs_array(session, args):
    """
    Firing rates as trials x units x bins, all units as run_anova, with the trial numbers,
    PseudoUnitIDs and bin start times in ms relative to the event along each axis
    """
    frs = io_utils.get_frs_from_args(args, session)
    trials = np.sort(frs.TrialNumber.unique())
    units = np.sort(frs.PseudoUnitID.unique())
    bins = np.sort(frs.TimeBins.unique())
    X = np.full((len(trials), len(units), len(bins)), np.nan)
    X[np.searchsorted(trials, frs.TrialNumber.values), np.searchsorted(units, frs.PseudoUnitID.values), np.searchsorted(bins, frs.TimeBins.values)] = frs.FiringRate.values
    if np.isnan(X).any():
        raise ValueError(f"missing firing rates for session {session}")
    bin_milli = np.round((bins - args.trial_interval.pre_interval / 1000) * 1000).astype(int)
    return trials, units, bin_milli, X


def get_windows(bin_milli, args):
    """
    Window starts and each window's bin indices, the windows of run_anova.process_session. One
    window of all bins without window_size
    """
    if args.window_size is None:
        return None, [np.arange(len(bin_milli))]
    ti = args.trial_interval
    starts = np.arange(-ti.pre_interval, ti.post_interval - args.window_size + 1, ti.interval_size)
    window_bins = [np.flatnonzero((bin_milli >= s) & (bin_milli < s + args.window_size)) for s in starts]
    if any(len(b) == 0 for b in window_bins):
        raise ValueError("a window has no time bins")
    return starts, window_bins


def run_fracvars(beh, trials, units, X, window_starts, window_bins, args):
    """
    run_anova.run_anova's result for one run of one session, rows ordered by window then unit.
    None if no trials
    """
    beh = beh[beh.TrialNumber.isin(trials)]
    if len(beh) == 0:
        return None
    idx = np.searchsorted(trials, beh.TrialNumber.values)
    cols = anova_utils.factorial_fracvar(X[idx], beh.reset_index(drop=True), args.conditions, window_bins)
    num_windows = len(window_bins)
    res = pd.DataFrame({"PseudoUnitID": np.tile(units, num_windows), **{k: v.T.ravel() for k, v in cols.items()}})
    res = anova_utils.combine_time_fracvar(res, args.conditions)
    if "BeliefPartition" in args.conditions and "BeliefConf" in args.conditions:
        res[f"x_BeliefPref_comb_time_fracvar"] = res[f"x_BeliefPartition_comb_time_fracvar"] - res[f"x_BeliefConf_comb_time_fracvar"]
    if "NextBeliefPartition" in args.conditions and "NextBeliefConf" in args.conditions:
        res[f"x_NextBeliefPref_comb_time_fracvar"] = res[f"x_NextBeliefPartition_comb_time_fracvar"] - res[f"x_NextBeliefConf_comb_time_fracvar"]
    res["feat"] = args.feat
    if window_starts is not None:
        res["WindowStartMilli"] = np.repeat(window_starts, len(units))
        res["WindowEndMilli"] = res["WindowStartMilli"] + args.window_size
    # run_anova's index: each window's units numbered from 0
    res.index = np.tile(np.arange(len(units)), num_windows)
    return res


def process_feat(args):
    feat_sessions = pd.read_pickle((args.feats_path or FEATS_PATH).format(sub=args.subject))
    valid_sess = pd.read_pickle(SESSIONS_PATH.format(sub=args.subject))
    feat_rows = feat_sessions[feat_sessions.feat == args.feat]
    if len(feat_rows) == 0 or len(feat_rows.iloc[0].sessions) == 0:
        print(f"No sessions for {args.feat}, skipping", flush=True)
        return
    sessions = valid_sess[valid_sess.session_name.isin(feat_rows.iloc[0].sessions)].session_name
    splits = pd.read_pickle(io_utils.get_anova_split_path(args)) if args.split_idx is not None else None
    print(f"Anova for {args.feat} using {len(sessions)} sessions, conditions {args.conditions}, {args.num_shuffles} shuffles", flush=True)

    shuffle_idxs = [None] + list(range(args.num_shuffles))
    runs = {k: [] for k in shuffle_idxs}
    dropped = set()
    for session in sessions:
        start = time.time()
        raw = behavioral_utils.get_valid_belief_beh_for_sub_sess(args.subject, session).reset_index(drop=True)
        labelled = label_beh(raw, args)
        split_trials = None
        if splits is not None:
            split_trials = splits[(splits.sessions == session) & (splits.feat == args.feat)].iloc[0][f"split_{args.split_idx}"]
        trials, units, bin_milli, X = load_frs_array(session, args)
        window_starts, window_bins = get_windows(bin_milli, args)
        for k in shuffle_idxs:
            if k in dropped:
                continue
            beh = shuffle_beh(raw, labelled, session, args, k, split_trials)
            res = run_fracvars(beh, trials, units, X, window_starts, window_bins, args)
            if res is None:
                if k is None:
                    raise ValueError(f"no data loaded for session {session}")
                print(f"shuffle {k}: no trials in session {session}, dropping the shuffle", flush=True)
                dropped.add(k)
                continue
            runs[k].append(res)
        print(f"{session}: {len(units)} units, {time.time() - start:.1f}s", flush=True)

    output_dir = io_utils.get_anova_output_dir(args)
    pd.concat(runs[None]).to_pickle(os.path.join(output_dir, f"{args.feat}_.pickle"))
    if args.num_shuffles == 0:
        return
    shuffles = []
    for k in shuffle_idxs[1:]:
        if k in dropped:
            continue
        res = pd.concat(runs[k])
        res["shuffle_idx"] = k
        shuffles.append(res)
    shuffles = pd.concat(shuffles)
    print(f"{shuffles.shuffle_idx.nunique()} shuffles for feat {args.feat}", flush=True)
    shuffles.to_pickle(os.path.join(output_dir, f"{args.feat}_shuffles.pickle"))
    keys = ["PseudoUnitID"] if args.window_size is None else ["PseudoUnitID", "WindowStartMilli"]
    stats = anova_utils.shuffle_stats(shuffles, args.conditions, keys)
    stats.to_pickle(os.path.join(output_dir, f"{args.feat}_shuffle_stats.pickle"))


def main():
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(AnovaConfigs(), parser)
    parser.add_argument("--num_shuffles", default=100, type=int)
    args = parser.parse_args()
    # results are written as the true run's; shuffles are indexed internally
    args.shuffle_idx = None
    args.trial_interval = get_trial_interval(args.trial_event)

    print(f"Using {args.fr_type} as inputs, time range {args.time_range}, filters {args.beh_filters}", flush=True)
    if args.b_split_half is not None:
        print(f"Keeping half B{args.b_split_half} of group B, split with seed {args.split_seed}", flush=True)
    feats = FEATURES if args.feat_idx is None else [FEATURES[args.feat_idx]]
    for feat in feats:
        args.feat = feat
        start = time.time()
        process_feat(args)
        print(f"{feat} done in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
