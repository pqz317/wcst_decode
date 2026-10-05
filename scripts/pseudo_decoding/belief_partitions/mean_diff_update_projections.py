"""
Do trial outcomes move pre-stimulus activity along feature X's preference and confidence axes the
way the Belief state model's beliefs move? Implements claude_notes/Mean-difference belief axes for
update projections proposal.md, replacing the decoder axes of pref_conf_projection_updates.py.

For each (session, feature X), every trial goes at random to the axis or the test half, stratified
by choice x outcome x partition. From the axis half, per time bin:

    pref = mean x | High X  -  mean x | High Not X
    conf = mean x | High    -  mean x | Low           (High = High X or High Not X)

Each chose-X test trial t contributes dx_t = x_{t+1} - x_t, whatever half t + 1 is in, projected per
bin and averaged over bins. When t + 1 is an axis trial this leaks; the white-noise runs measure
how much.

--split pair instead draws half of each chose-X cell (outcome x partition) as test trials t,
reserves t and t + 1, and puts every other trial, not-X trials included, in the axis. No trial is
then in both the axis and a test change.

Hypotheses, as within-(session, feature) contrasts of chose-X cells signed to predict > 0, tested
by sign flips per (session, feature):

    conf_cor:  cor/Low - cor/High
    conf_inc:  inc/Low - inc/High
    pref_cor:  cor/Low - cor/High X
    pref_inc:  inc/High Not X - inc/High X

--frs_source white_noise replaces the firing rates with N(0, 1) of each session's real
(trials x units x bins) shape, seeded by --noise_seed. The split depends only on --train_test_seed,
so it's identical to the neural run's.

--sig_units restricts each feature's pref axis to its {pref}_99th_no_cond_window_filter_drift units
and its conf axis to its {conf}_99th_no_cond_window_filter_drift units, the sets the old decoder
axes used. A (session, feature) with no such units for an axis is left out of that axis's stats.

One process does both monkeys; no slurm launcher is needed.
"""

import os
import copy
import argparse
import numpy as np
import pandas as pd

import utils.behavioral_utils as behavioral_utils
import utils.stats_utils as stats_utils
from constants.behavioral_constants import *
from constants.decoding_constants import *
from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import BeliefPartitionConfigs, add_defaults_to_parser
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
from scripts.pseudo_decoding.belief_partitions.stim_belief_vector_alignment import get_feat_to_sessions, load_session_frs
from scripts.pseudo_decoding.belief_partitions.prior_dependent_updates import load_beh, PRE_STIM_RANGE

MODE = "mean_diff_updates"
OUTPUT_PATH = "/data/patrick_res/mean_diff_update_projections"

FRS_SOURCES = ["neural", "white_noise"]

OUTCOMES = {"Correct": "cor", "Incorrect": "inc"}
PARTITIONS = ["Low", "High X", "High Not X"]
AXES = ["pref", "conf"]
SIG_UNIT_LEVEL = "{axis}_99th_no_cond_window_filter_drift"

# strata of the split, as (chose X, outcome, partition); the index seeds each stratum's draw
STRATA = [(c, o, p) for c in [True, False] for o in ["cor", "inc"] for p in PARTITIONS]

PREF_CELLS = [f"{o}/{p}" for o in ["cor", "inc"] for p in PARTITIONS]
CONF_CELLS = [f"{o}/{c}" for o in ["cor", "inc"] for c in ["Low", "High"]]
AXIS_CELLS = {"pref": PREF_CELLS, "conf": CONF_CELLS}

# contrasts as (axis, cell summed with +, cell summed with -), over chose-X test events
CONTRASTS = {
    "conf_cor": ("conf", "cor/Low", "cor/High"),
    "conf_inc": ("conf", "inc/Low", "inc/High"),
    "pref_cor": ("pref", "cor/Low", "cor/High X"),
    "pref_inc": ("pref", "inc/High Not X", "inc/High X"),
}


def session_activity(sess_name, beh, args):
    """
    (trials x units x bins activity, TrialNumber -> row, ascending PseudoUnitIDs) from the source
    --frs_source names, or None if the session has no firing rates.

    Each (unit, bin) is divided by its sd over the session's valid trials. This scaling sees no
    labels, so it can't create differences between cells. Zero-sd entries go to 0, as in
    spike_utils.zscore_frs.
    """
    session_frs = load_session_frs(sess_name, args)
    if session_frs is None:
        return None
    X, trial_pos, unit_ids = session_frs
    if args.frs_source == "white_noise":
        rng = np.random.default_rng([int(sess_name), args.noise_seed])
        X = rng.standard_normal(X.shape)
    valid = [trial_pos[t] for t in beh.TrialNumber if t in trial_pos]
    sd = X[valid].std(axis=0, ddof=1)
    inv_sd = np.divide(1.0, sd, out=np.zeros_like(sd), where=sd > 0)
    return X * inv_sd, trial_pos, unit_ids


def load_sig_units(args):
    """
    {axis: DataFrame of feat, PseudoUnitID} for this subject, or None without --sig_units.
    """
    if not args.sig_units:
        return None
    return {
        a: pd.read_pickle(SIG_UNITS_PATH.format(sub=args.subject, event=args.trial_event, level=SIG_UNIT_LEVEL.format(axis=a)))
        for a in AXES
    }


def axis_unit_masks(unit_ids, feat, sig):
    """
    {axis: boolean mask over unit_ids} of the units each axis may use; all True without --sig_units.
    """
    if sig is None:
        return {a: np.ones(len(unit_ids), dtype=bool) for a in AXES}
    return {a: np.isin(unit_ids, sig[a][sig[a].feat == feat].PseudoUnitID) for a in AXES}


def label_trials(beh, feat):
    """
    Per trial: outcome, whether feat was chosen, belief partition relative to feat on that trial,
    the next trial's number, and the chose-X pref and conf cells ("not X" for not-chose-X trials).
    """
    beh = behavioral_utils.get_feat_choice_label(beh.copy(), feat)
    beh = behavioral_utils.get_belief_partitions(beh, feat, use_x=True)
    trials = pd.DataFrame({
        "TrialNumber": beh.TrialNumber.to_numpy(),
        "NextTrialNumber": beh.NextTrialNumber.to_numpy(),
        "outcome": beh.Response.map(OUTCOMES).to_numpy(),
        "partition": beh.BeliefPartition.to_numpy(),
        "chose": (beh.Choice == "Chose").to_numpy(),
    })
    conf_level = np.where(trials.partition == "Low", "Low", "High")
    trials["pref_cell"] = np.where(trials.chose, trials.outcome + "/" + trials.partition, "not X")
    trials["conf_cell"] = np.where(trials.chose, trials.outcome + "/" + conf_level, "not X")
    return trials


def split_halves(sess_name, feat, trials, seed):
    """
    Adds `half`: each stratum (chose x outcome x partition) is split in half, the odd trial of an
    odd-sized stratum going to a random half.

    Drawn over behavioral trial numbers with --train_test_seed, so the split is identical whatever
    --frs_source or --noise_seed is, and deterministic in (session, feat, seed, stratum).
    """
    trials["half"] = "axis"
    for code, (chose, outcome, partition) in enumerate(STRATA):
        in_stratum = (trials.chose == chose) & (trials.outcome == outcome) & (trials.partition == partition)
        stratum = np.sort(trials.loc[in_stratum, "TrialNumber"].to_numpy())
        rng = np.random.default_rng([int(sess_name), FEATURES.index(feat), seed, code])
        perm = rng.permutation(stratum)
        n_test = len(perm) // 2 + rng.integers(len(perm) % 2 + 1)
        trials.loc[trials.TrialNumber.isin(perm[:n_test]), "half"] = "test"
        # each stratum splits within one trial of even
        n_axis = len(perm) - n_test
        assert abs(n_test - n_axis) <= 1
    return trials


def split_pairs(sess_name, feat, trials, seed):
    """
    Adds `half`: in each chose-X cell, half of the trials that have a next trial are drawn as
    "test", the odd one going to a random half; each test trial's next trial, if not itself a test
    trial, is "reserved"; every other trial is "axis". Seeded like split_halves, per cell.
    """
    trials["half"] = "axis"
    for code, cell in enumerate(PREF_CELLS):
        in_cell = (trials.pref_cell == cell) & (trials.NextTrialNumber >= 0)
        cell_trials = np.sort(trials.loc[in_cell, "TrialNumber"].to_numpy())
        rng = np.random.default_rng([int(sess_name), FEATURES.index(feat), seed, code])
        perm = rng.permutation(cell_trials)
        n_test = len(perm) // 2 + rng.integers(len(perm) % 2 + 1)
        trials.loc[trials.TrialNumber.isin(perm[:n_test]), "half"] = "test"
    test_next = trials.loc[trials.half == "test", "NextTrialNumber"]
    trials.loc[trials.TrialNumber.isin(test_next) & (trials.half != "test"), "half"] = "reserved"
    # no trial of any test pair is an axis trial
    axis_trials = set(trials.loc[trials.half == "axis", "TrialNumber"])
    assert axis_trials.isdisjoint(set(trials.loc[trials.half == "test", "TrialNumber"]) | set(test_next))
    return trials


def feature_axes(X, rows, trials):
    """
    Pref and conf axes (each units x bins) from the axis half, as ({axis: u}, {group: n}), or
    (None, counts) if any group is empty.
    """
    axis = trials[(trials.half == "axis") & trials.TrialNumber.isin(list(rows))]
    groups = {
        "high_x": axis.partition == "High X",
        "high_not_x": axis.partition == "High Not X",
        "high": axis.partition.isin(["High X", "High Not X"]),
        "low": axis.partition == "Low",
    }
    means, counts = {}, {}
    for name, mask in groups.items():
        idx = [rows[t] for t in axis[mask].TrialNumber]
        counts[f"n_axis_{name}"] = len(idx)
        if len(idx) == 0:
            return None, counts
        means[name] = X[idx].mean(axis=0)
    axes = {"pref": means["high_x"] - means["high_not_x"], "conf": means["high"] - means["low"]}
    return axes, counts


def project_tests(X, rows, trials, axes):
    """
    The chose-X test trials whose activity on t and t + 1 both exist, with each axis's projection
    of dx_t, per bin then averaged over bins. Unnormalized: the norm needs every session.
    """
    test = trials[
        (trials.half == "test") & trials.chose
        & trials.TrialNumber.isin(list(rows)) & trials.NextTrialNumber.isin(list(rows))
    ].copy()
    cur = [rows[t] for t in test.TrialNumber]
    nxt = [rows[t] for t in test.NextTrialNumber]
    dX = X[nxt] - X[cur]
    for name, u in axes.items():
        test[f"proj_{name}"] = np.einsum("nub,ub->nb", dX, u).mean(axis=1)
    return test


def process_session(sess_name, feats, args, sig):
    """
    The test events' projections and the axis summaries for every valid feature of one session.
    Behavior and activity are read once and shared across features. Each axis is zeroed outside
    its allowed units, which makes the projection and norm those of the subset.
    """
    beh = load_beh(sess_name, args)
    activity = session_activity(sess_name, beh, args)
    if activity is None:
        return [], []
    X, rows, unit_ids = activity

    events_res, axes_res = [], []
    for feat in feats:
        split = split_pairs if args.split == "pair" else split_halves
        trials = split(sess_name, feat, label_trials(beh, feat), args.train_test_seed)
        assert trials.TrialNumber.is_unique
        axes, counts = feature_axes(X, rows, trials)
        masks = axis_unit_masks(unit_ids, feat, sig)
        axis_row = {"session": sess_name, "subject": args.subject, "feat": feat, "n_units": X.shape[1],
                    **{f"n_units_{a}": int(m.sum()) for a, m in masks.items()}, **counts}
        if axes is None:
            print(f"session {sess_name} feat {feat}: an axis group is empty, dropping", flush=True)
            axes_res.append({**axis_row, **{f"sq_norm_{a}": np.nan for a in AXES}, "dropped": True})
            continue
        axes = {a: u * masks[a][:, None] for a, u in axes.items()}
        axes_res.append({**axis_row, **{f"sq_norm_{a}": float((u ** 2).sum()) for a, u in axes.items()}, "dropped": False})

        test = project_tests(X, rows, trials, axes)
        for a, m in masks.items():
            if not m.any():
                test[f"proj_{a}"] = np.nan
        test["session"] = sess_name
        test["subject"] = args.subject
        test["feat"] = feat
        events_res.append(test[[
            "session", "subject", "feat", "TrialNumber", "outcome", "partition",
            "pref_cell", "conf_cell", "proj_pref", "proj_conf",
        ]])
    return events_res, axes_res


def compute_stats(events, axes, num_flips=10000, seed=42):
    """
    Sign-flip tests of CONTRASTS, plus per-cell summaries.

    Each axis's projections are divided by its pseudo-population norm per feature,
    sqrt(sum over sessions and bins of ||u_s||^2), so each feature's axis has unit length across
    all bins. A contrast d_{s,X} is formed for each (session, feat) with both cells non-empty, and
    D = sum d_{s,X} is tested by flipping signs per (session, feat). Conf's High cell is the mean
    over every test trial in High X or High Not X.

    Returns (stats, cells):
      stats: one row per contrast -- D, mean_d, n_units, p
      cells: per axis and cell, the mean and SE across (session, feat) units of the cell mean,
             n_units, n_events
    """
    kept = axes[~axes.dropped]
    events = events.copy()
    unit_means = {}
    for a in AXES:
        norms = np.sqrt(kept.groupby("feat")[f"sq_norm_{a}"].sum())
        events[f"proj_{a}_n"] = events[f"proj_{a}"] / events.feat.map(norms)
        unit_means[a] = (
            events.groupby(["session", "feat", f"{a}_cell"])[f"proj_{a}_n"].mean()
            .unstack(f"{a}_cell").reindex(columns=AXIS_CELLS[a])
        )

    rng = np.random.default_rng(seed)
    stats = []
    for name, (a, plus, minus) in CONTRASTS.items():
        d = (unit_means[a][plus] - unit_means[a][minus]).dropna()
        D, p = stats_utils.sign_flip_test(d.to_numpy(), num_flips, rng)
        stats.append({"contrast": name, "axis": a, "plus": plus, "minus": minus,
                      "D": D, "mean_d": d.mean(), "n_units": len(d), "p": p})

    cells = []
    for a in AXES:
        for cell in AXIS_CELLS[a]:
            vals = unit_means[a][cell].dropna()
            cells.append({
                "axis": a, "cell": cell, "mean": vals.mean(), "se": vals.std(ddof=1) / np.sqrt(len(vals)),
                "n_units": len(vals), "n_events": int(events.loc[events[f"{a}_cell"] == cell, f"proj_{a}"].notna().sum()),
            })
    return pd.DataFrame(stats), pd.DataFrame(cells)


def file_prefix(args):
    seed_str = "" if args.train_test_seed == 42 else f"_seed_{args.train_test_seed}"
    noise_str = f"_noise_{args.noise_seed}" if args.frs_source == "white_noise" else ""
    sig_str = "_sig_units" if args.sig_units else ""
    split_str = "_pair_split" if args.split == "pair" else ""
    return f"{MODE}_{args.frs_source}{noise_str}{sig_str}{split_str}{seed_str}"


def print_counts(events):
    per_session = events.groupby(["session", "pref_cell"]).size().unstack("pref_cell").fillna(0)
    print("\nchose-X test events per session, pooled over features:", flush=True)
    print(pd.DataFrame({
        "median": per_session.median(), "min": per_session.min(), "sessions with 0": (per_session == 0).sum(),
    }).reindex(PREF_CELLS).to_string(), flush=True)


def main(args):
    if args.frs_source == "white_noise" and args.noise_seed is None:
        raise ValueError("--frs_source white_noise needs --noise_seed")
    args.trial_interval = get_trial_interval(args.trial_event)
    args.time_range = PRE_STIM_RANGE
    subjects = ["SA", "BL"] if args.subject == "both" else [args.subject]

    events, axes = [], []
    for sub in subjects:
        sub_args = copy.deepcopy(args)
        sub_args.subject = sub
        feat_to_sessions, sub_sessions = get_feat_to_sessions(sub_args)
        sig = load_sig_units(sub_args)
        for sess_name in sub_sessions.session_name:
            feats = [f for f in FEATURES if sess_name in feat_to_sessions[f]]
            print(f"{sub} session {sess_name}: {len(feats)} valid features", flush=True)
            sess_events, sess_axes = process_session(sess_name, feats, sub_args, sig)
            events.extend(sess_events)
            axes.extend(sess_axes)
    events = pd.concat(events, ignore_index=True)
    axes = pd.DataFrame(axes)
    stats, cells = compute_stats(events, axes, args.num_flips, args.train_test_seed)

    output_dir = belief_partitions_io.get_dir_name(args)
    prefix = file_prefix(args)
    for name, df in [("events", events), ("axes", axes), ("stats", stats), ("cells", cells)]:
        df.to_pickle(os.path.join(output_dir, f"{prefix}_{name}.pickle"))

    print(f"\nsaved {prefix}_*.pickle to {output_dir}", flush=True)
    print(f"{axes.session.nunique()} sessions, {(~axes.dropped).sum()} (session, feat) units, "
          f"{axes.dropped.sum()} dropped", flush=True)
    print_counts(events)
    print("\ncells:", flush=True)
    print(cells.to_string(), flush=True)
    print("\nstats:", flush=True)
    print(stats.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(
        BeliefPartitionConfigs(mode=MODE, subject="both", base_output_path=OUTPUT_PATH), parser
    )
    parser.add_argument("--frs_source", default="neural", choices=FRS_SOURCES)
    parser.add_argument("--noise_seed", default=None, type=int)
    parser.add_argument("--num_flips", default=10000, type=int)
    parser.add_argument("--sig_units", action="store_true")
    parser.add_argument("--split", default="half", choices=["half", "pair"])
    main(parser.parse_args())
