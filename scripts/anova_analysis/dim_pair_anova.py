"""
Units selective for a pair of feature dimensions, for the held-out dimension CCGP
(dim_ccgp_heldout_feat_beliefs.py). The dimension counterpart of the
pref_99th_no_cond_window_filter_drift units of run_anova.py -> 20250329_aggregate_shuffles.py ->
20250606_generate_significant_units.py.

For each dimension pair of generate_dim_heldout_sessions.py and each of its sessions, the session's
six features are the ones the decoder's true run uses (dim_ccgp_heldout_feat_beliefs.assign_roles).
High trials of each (PreferredBelief == f, PreferredBeliefProb > BELIEF_PARTITION_THRESH) are
balanced to the minimum count over the six and labelled by dimension. Per unit and 500 ms window,
sliding by 100 ms within [-1, 0]s of StimOnset, the statistic is the fraction of variance explained
by Dimension and Dimension x TimeBins, i.e. run_anova's x_Dimension_comb_time_fracvar with conditions
[TimeBins, Dimension], computed from cell means by anova_utils.cell_fracvar.

The null is NUM_SHUFFLES circular shifts of the session's behavior, shuffle_beh_by_shift with buffer
50 and seed k, as run_anova does with --shuffle_method circular_shift. The features stay those of the
true behavior; a rotation keeps every trial, so their counts are unchanged. A unit is selective for
the pair if its statistic exceeds its own 99th shuffle percentile in any window.

Writes, per subject:
    OUTPUT_PATH/{sub}_StimOnset_Dimension_window_500_pre_stim/{dim_pair}.pickle
        one row per (unit, window): statistic, total variance, shuffle quantiles, significance, and
        the shuffle values themselves
    SIG_UNITS_PATH with level SIG_LEVEL
        the selective units, columns dim_pair, structure_level2, structure_level2_cleaned,
        PseudoUnitID, session

    python3 scripts/anova_analysis/dim_pair_anova.py
"""

import os
import argparse
import numpy as np
import pandas as pd

import utils.behavioral_utils as behavioral_utils
import utils.spike_utils as spike_utils
import utils.anova_utils as anova_utils
from constants.behavioral_constants import *
from constants.decoding_constants import *

from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import BeliefPartitionConfigs
from scripts.pseudo_decoding.belief_partitions.generate_dim_heldout_sessions import DIM_PAIRS_PATH
from scripts.pseudo_decoding.belief_partitions.dim_ccgp_heldout_feat_beliefs import assign_roles, is_qualifying

OUTPUT_PATH = "/data/patrick_res/anova/dim_pairs"
RUN_NAME = "{sub}_StimOnset_Dimension_window_500_pre_stim"
SIG_LEVEL = "dim_pair_99th_pre_stim_window_filter_drift"
TRIAL_EVENT = "StimOnset"
PRE_STIM_RANGE = [-1000, 0]
WINDOW_SIZE = 500
SHUFFLE_BUFFER = 50


def get_windows(trial_interval):
    """
    Window starts in ms, and each window's bin indices into the PRE_STIM_RANGE bins
    """
    step = trial_interval.interval_size
    starts = np.arange(PRE_STIM_RANGE[0], PRE_STIM_RANGE[1] - WINDOW_SIZE + 1, step)
    window_bins = [np.arange((s - PRE_STIM_RANGE[0]) // step, (s - PRE_STIM_RANGE[0] + WINDOW_SIZE) // step) for s in starts]
    return starts, window_bins


def load_frs_array(sess_name, args):
    """
    Firing rates as trials x units x bins, with the trial numbers and PseudoUnitIDs along each axis.
    Units are all of the session's, minus drifting and bad-region units
    """
    frs = spike_utils.get_frs_from_args(args, sess_name)
    trials = np.sort(frs.TrialNumber.unique())
    units = np.sort(frs.PseudoUnitID.unique())
    bin_idx = np.round((frs.TimeBins.values - PRE_STIM_RANGE[0] / 1000 - args.trial_interval.pre_interval / 1000) * 10).astype(int)
    X = np.full((len(trials), len(units), bin_idx.max() + 1), np.nan)
    X[np.searchsorted(trials, frs.TrialNumber.values), np.searchsorted(units, frs.PseudoUnitID.values), bin_idx] = frs.FiringRate.values
    if np.isnan(X).any():
        raise ValueError(f"missing firing rates for session {sess_name}")
    return trials, units, X


def select_balanced(beh, feat_to_dim, dim_a, rng):
    """
    High trials of the features in feat_to_dim, each subsampled to the minimum count over them.
    Returns their trial numbers and dimension index (0 for dim_a, 1 otherwise)
    """
    high = beh[(beh.PreferredBeliefProb > BELIEF_PARTITION_THRESH) & beh.PreferredBelief.isin(feat_to_dim)]
    by_feat = {f: np.sort(high[high.PreferredBelief == f].TrialNumber.values) for f in feat_to_dim}
    min_count = min(len(t) for t in by_feat.values())
    trials, dims = [], []
    for f, t in by_feat.items():
        trials.append(rng.choice(t, min_count, replace=False))
        dims.append(np.full(min_count, 0 if feat_to_dim[f] == dim_a else 1))
    return np.concatenate(trials), np.concatenate(dims), min_count


def session_fracvars(sess_name, args, window_starts, window_bins):
    """
    Per unit and window of a session: the true statistic and the num_shuffles circular shift values
    """
    beh = behavioral_utils.get_valid_belief_beh_for_sub_sess(args.subject, sess_name)
    if not is_qualifying(beh, args):
        print(f"Skipping session {sess_name}, not three qualifying features per dimension")
        return None
    role_to_feat = assign_roles(beh, sess_name, args)
    feat_to_dim = {f: FEATURE_TO_DIM[f] for f in role_to_feat.values()}
    trial_nums, units, X = load_frs_array(sess_name, args)

    runs = []
    for k in [None] + list(range(args.num_shuffles)):
        run_beh = beh if k is None else behavioral_utils.shuffle_beh_by_shift(beh.copy(), buffer=SHUFFLE_BUFFER, seed=k)
        rng = np.random.default_rng([args.train_test_seed, int(sess_name), 0 if k is None else k + 1])
        trials, dims, min_count = select_balanced(run_beh, feat_to_dim, args.dim_a, rng)
        keep = np.isin(trials, trial_nums)
        idx = np.searchsorted(trial_nums, trials[keep])
        fracvar, total_var = anova_utils.cell_fracvar(X[idx], dims[keep], window_bins)
        runs.append((fracvar, total_var))

    true_fracvar, true_total_var = runs[0]
    shuffles = np.stack([fracvar for fracvar, _ in runs[1:]], axis=-1)
    q95, q99 = np.quantile(shuffles, 0.95, axis=-1), np.quantile(shuffles, 0.99, axis=-1)
    p = np.select([true_fracvar > q99, true_fracvar > q95], ["99th", "95th"], "insignificant")

    unit_grid, window_grid = np.meshgrid(np.arange(len(units)), np.arange(len(window_starts)), indexing="ij")
    res = pd.DataFrame({
        "PseudoUnitID": units[unit_grid.ravel()],
        "session": sess_name,
        "WindowStartMilli": window_starts[window_grid.ravel()],
        "WindowEndMilli": window_starts[window_grid.ravel()] + WINDOW_SIZE,
        "x_Dimension_comb_time_fracvar": true_fracvar.ravel(),
        "total_var": true_total_var.ravel(),
        "Dimension_95th": q95.ravel(),
        "Dimension_99th": q99.ravel(),
        "p": p.ravel(),
        "n_trials_per_feat": min_count,
        "feats": ["_".join(role_to_feat.values())] * unit_grid.size,
    })
    res["shuffle_fracvars"] = list(shuffles.reshape(-1, shuffles.shape[-1]))
    return res


def run_sub(sub, dim_pairs, args):
    args.subject = sub
    window_starts, window_bins = get_windows(args.trial_interval)
    output_dir = os.path.join(args.base_output_path, RUN_NAME.format(sub=sub))
    os.makedirs(output_dir, exist_ok=True)

    sig_units = []
    for _, row in dim_pairs.iterrows():
        args.dim_a, args.dim_b = row.dim_a, row.dim_b
        dim_pair = f"{row.dim_a}_{row.dim_b}"
        sessions = [s for s in row.sessions if behavioral_utils.get_sub_for_session(s) == sub]
        print(f"{sub} {dim_pair}: {len(sessions)} sessions", flush=True)
        res = [session_fracvars(s, args, window_starts, window_bins) for s in sessions]
        res = [r for r in res if r is not None]
        if len(res) == 0:
            continue
        res = pd.concat(res, ignore_index=True)
        res["dim_pair"] = dim_pair
        res.to_pickle(os.path.join(output_dir, f"{dim_pair}.pickle"))

        sig = res[res.p == "99th"].PseudoUnitID.unique()
        print(f"{sub} {dim_pair}: {len(sig)} of {res.PseudoUnitID.nunique()} units selective", flush=True)
        sig_units.append(pd.DataFrame({"dim_pair": dim_pair, "PseudoUnitID": sig}))

    sig_units = pd.concat(sig_units, ignore_index=True)
    positions = spike_utils.get_good_subject_units(sub)[["PseudoUnitID", "structure_level2", "structure_level2_cleaned"]]
    sig_units = pd.merge(sig_units, positions, on="PseudoUnitID")
    sig_units["session"] = (sig_units.PseudoUnitID // 100).astype(int)
    sig_units = sig_units[["dim_pair", "structure_level2", "structure_level2_cleaned", "PseudoUnitID", "session"]]
    if not args.dry_run:
        sig_units.to_pickle(SIG_UNITS_PATH.format(sub=sub, event=TRIAL_EVENT, level=SIG_LEVEL))
    return sig_units


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_shuffles", default=100, type=int)
    parser.add_argument("--base_output_path", default=OUTPUT_PATH)
    parser.add_argument("--dry_run", action="store_true", help="skip writing the unit lists")
    cli = parser.parse_args()

    args = argparse.Namespace(**BeliefPartitionConfigs()._asdict())
    args.trial_event = TRIAL_EVENT
    args.trial_interval = get_trial_interval(TRIAL_EVENT)
    args.time_range = PRE_STIM_RANGE
    args.sig_unit_level = None
    args.num_shuffles = cli.num_shuffles
    args.base_output_path = cli.base_output_path
    args.dry_run = cli.dry_run

    dim_pairs = pd.read_pickle(DIM_PAIRS_PATH)
    for sub in ["SA", "BL"]:
        sig_units = run_sub(sub, dim_pairs, args)
        print(sig_units.groupby("dim_pair").PseudoUnitID.nunique())


if __name__ == "__main__":
    main()
