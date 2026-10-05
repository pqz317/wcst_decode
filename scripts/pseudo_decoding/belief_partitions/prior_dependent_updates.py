"""
Does the trial-to-trial change in pre-stimulus activity depend on the prior belief, the way the
Belief state model's update does? Implements claude_notes/Prior-dependent belief update analysis
proposal.md.

For each (session, feature X):

    dz_k = scaled ( x_{k+1} - x_k ),    x = pre-stimulus firing rates, averaged over the window

    u_X = (mean dz | X chosen, cor - mean dz | X chosen, inc)
        - (mean dz | X not chosen, cor - mean dz | X not chosen, inc)

built from half of the chose-X trials plus every not-X trial. The other half of the chose-X trials
is projected onto u_X and sorted into six cells, outcome x belief partition on trial k:

    {cor, inc} x {Low, High X, High Not X}

High Not X keeps only trials where the dominant feature W was on the chosen card. Chose-X trials
with W off the card are never tested, so they all go to the axis.

Hypotheses, as within-(session, feature) contrasts summed over units and tested by sign flips per
(session, feature) unit:

    H1:      P(cor/Low) - P(cor/High X)        > 0
    H2:      P(inc/High Not X) - P(inc/High X) > 0
    control: P(cor) - P(inc), pooled over partitions, > 0. Every not-X trial is in the axis, so
             the held-out choice x outcome interaction can't be formed; this is the held-out
             outcome contrast within chose X instead. Its null value is 0 if u_X carries no signal.

There is no shuffle: the axis never sees a test trial, so each contrast's null value is 0.

--frs_source swaps what x is, for verifying the pipeline:
    neural       the real firing rates
    beliefs      the 12 belief-state probabilities, as 12 units; should reproduce the ordering of
                 claude_notes/prior_dependent_update_predictions.py
    white_noise  N(0, 1) with each session's real (trials x units) shape; every p should be ~uniform

One process does both monkeys in a few minutes; no slurm launcher is needed.
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

MODE = "prior_dep_updates"
OUTPUT_PATH = "/data/patrick_res/prior_dependent_updates"

# pre-stimulus window, in ms relative to StimOnset, in the convention of spike_utils.get_frs_from_args
PRE_STIM_RANGE = [-1000, 0]

FRS_SOURCES = ["neural", "beliefs", "white_noise"]

OUTCOMES = {"Correct": "cor", "Incorrect": "inc"}
PARTITIONS = ["Low", "High X", "High Not X"]
TEST_CELLS = [f"{o}/{p}" for o in ["cor", "inc"] for p in PARTITIONS]
# a seed field, so each cell's half-split is an independent draw
CELL_CODE = {cell: i for i, cell in enumerate(TEST_CELLS)}
UNTESTED = "untested"
NOT_X = "not X"

# axis cells, as (chose X, outcome)
AXIS_CELLS = [(True, "cor"), (True, "inc"), (False, "cor"), (False, "inc")]

# contrasts as (cells summed with +, cells summed with -), over a (session, feat)'s test events.
# The control pools each outcome over all three partitions
CONTRASTS = {
    "H1": ("cor/Low", "cor/High X"),
    "H2": ("inc/High Not X", "inc/High X"),
    "control": ("cor", "inc"),
}
HOLM_FAMILY = ["H1", "H2"]


def load_beh(sess_name, args):
    """
    Valid behavior with the next valid trial attached. NextTrialNumber is computed before any
    filtering, so k + 1 is the next valid trial, as in the belief model's trial sequence.
    """
    beh = behavioral_utils.load_behavior_from_args(sess_name, args)
    beh = beh.sort_values("TrialNumber").reset_index(drop=True)
    # -1 marks the last trial, which has no next and so never forms a pair
    beh["NextTrialNumber"] = beh.TrialNumber.shift(-1).fillna(-1).astype(int)
    return beh


def session_activity(sess_name, beh, args):
    """
    (trials x units activity, TrialNumber -> row), averaged over the window's time bins, from the
    source --frs_source names. None if the session has no firing rates.
    """
    if args.frs_source == "beliefs":
        X = beh[[f"{feat}Prob" for feat in FEATURES]].to_numpy(dtype=float)
        return X, {t: i for i, t in enumerate(beh.TrialNumber)}

    session_frs = load_session_frs(sess_name, args)
    if session_frs is None:
        return None
    X, trial_pos, _ = session_frs
    X = X.mean(axis=2)
    if args.frs_source == "white_noise":
        rng = np.random.default_rng([int(sess_name), args.train_test_seed])
        X = rng.standard_normal(X.shape)
    return X, trial_pos


def session_changes(beh, X, trial_pos):
    """
    dz_k for every trial k whose activity on both k and k + 1 exists, as (dZ, TrialNumber -> row).

    Each unit is divided by the sd of its change over all of the session's pairs, a scaling that
    sees no labels, so including the test trials biases no contrast. Zero-sd units go to 0, as
    in spike_utils.zscore_frs.
    """
    has_pair = beh.TrialNumber.isin(list(trial_pos)) & beh.NextTrialNumber.isin(list(trial_pos))
    pairs = beh[has_pair]
    cur = [trial_pos[t] for t in pairs.TrialNumber]
    nxt = [trial_pos[t] for t in pairs.NextTrialNumber]
    dX = X[nxt] - X[cur]
    sd = dX.std(axis=0, ddof=1)
    inv_sd = np.divide(1.0, sd, out=np.zeros_like(sd), where=sd > 0)
    return dX * inv_sd, {t: i for i, t in enumerate(pairs.TrialNumber)}


def label_events(beh, feat):
    """
    Per trial: outcome, belief partition relative to feat, whether feat was chosen, and the cell
    the trial belongs to -- one of TEST_CELLS, UNTESTED (chose X while High Not X, W off the card)
    or NOT_X.
    """
    beh = behavioral_utils.get_feat_choice_label(beh.copy(), feat)
    beh = behavioral_utils.get_belief_partitions(beh, feat, use_x=True)
    events = pd.DataFrame({
        "TrialNumber": beh.TrialNumber.to_numpy(),
        "outcome": beh.Response.map(OUTCOMES).to_numpy(),
        "partition": beh.BeliefPartition.to_numpy(),
        "chose": (beh.Choice == "Chose").to_numpy(),
    })
    w_off_card = (beh.BeliefPartition == "High Not X").to_numpy() & ~beh.PreferredChosen.to_numpy(dtype=bool)
    events["cell"] = np.where(
        ~events.chose, NOT_X,
        np.where(w_off_card, UNTESTED, events.outcome + "/" + events.partition)
    )
    return events


def split_halves(sess_name, feat, events, seed):
    """
    Adds `half`: each test cell's trials are split in half, the odd trial of an odd-sized cell
    going to a random half. NOT_X and UNTESTED trials all go to the axis.

    Drawn over behavioral trial numbers, so the split is identical whatever --frs_source is, and
    deterministic in (session, feat, seed, cell).
    """
    events["half"] = "axis"
    for cell in TEST_CELLS:
        trials = np.sort(events.loc[events.cell == cell, "TrialNumber"].to_numpy())
        rng = np.random.default_rng([int(sess_name), FEATURES.index(feat), seed, CELL_CODE[cell]])
        perm = rng.permutation(trials)
        n_test = len(perm) // 2 + rng.integers(len(perm) % 2 + 1)
        events.loc[events.TrialNumber.isin(perm[:n_test]), "half"] = "test"
    return events


def feature_axis(dZ, rows, events):
    """
    u_X from the axis half, as (u, {axis cell: n}), or (None, counts) if any axis cell is empty.
    """
    axis = events[events.half == "axis"]
    means, counts = {}, {}
    for chose, outcome in AXIS_CELLS:
        trials = axis[(axis.chose == chose) & (axis.outcome == outcome)].TrialNumber
        idx = [rows[t] for t in trials if t in rows]
        counts[f"n_axis_{'chose' if chose else 'not'}_{outcome}"] = len(idx)
        if len(idx) == 0:
            return None, counts
        means[(chose, outcome)] = dZ[idx].mean(axis=0)
    u = (means[(True, "cor")] - means[(True, "inc")]) - (means[(False, "cor")] - means[(False, "inc")])
    return u, counts


def process_session(sess_name, feats, args):
    """
    The test events' projections and the axis summaries for every valid feature of one session.
    Behavior and activity are read once and shared across features.
    """
    beh = load_beh(sess_name, args)
    activity = session_activity(sess_name, beh, args)
    if activity is None:
        return [], []
    dZ, rows = session_changes(beh, *activity)

    events_res, axes_res = [], []
    for feat in feats:
        events = split_halves(sess_name, feat, label_events(beh, feat), args.train_test_seed)
        u, counts = feature_axis(dZ, rows, events)
        axis_row = {"session": sess_name, "subject": args.subject, "feat": feat, "n_units": dZ.shape[1], **counts}
        if u is None:
            print(f"session {sess_name} feat {feat}: an axis cell is empty, dropping", flush=True)
            axes_res.append({**axis_row, "sq_norm": np.nan, "dropped": True})
            continue
        axes_res.append({**axis_row, "sq_norm": float(u @ u), "dropped": False})

        test = events[(events.half == "test") & events.TrialNumber.isin(list(rows))].copy()
        test["proj"] = dZ[[rows[t] for t in test.TrialNumber]] @ u
        test["session"] = sess_name
        test["subject"] = args.subject
        test["feat"] = feat
        events_res.append(test[["session", "subject", "feat", "TrialNumber", "outcome", "partition", "cell", "proj"]])
    return events_res, axes_res


def compute_stats(events, axes, num_flips=10000, seed=42):
    """
    Sign-flip tests of CONTRASTS, plus per-cell summaries for plotting.

    Projections are divided by each feature's pseudo-population axis norm, sqrt(sum_s ||u_X,s||^2),
    so each feature's axis has unit length. A contrast d_{s,X} is formed for each (session, feat)
    with both sides non-empty, and D = sum d_{s,X} is tested by flipping signs per (session, feat).
    That treats features within a session as independent, a simplification the proposal accepts.

    Returns (stats, cells):
      stats: one row per contrast -- D, mean_d, n_units, p, and p_holm over HOLM_FAMILY
      cells: per cell, the mean and SE across (session, feat) units of the cell mean, n_units, n_events
    """
    norms = np.sqrt(axes[~axes.dropped].groupby("feat").sq_norm.sum())
    events = events.copy()
    events["proj_n"] = events.proj / events.feat.map(norms)

    unit_means = events.groupby(["session", "feat", "cell"]).proj_n.mean().unstack("cell").reindex(columns=TEST_CELLS)
    unit_means = unit_means.join(events.groupby(["session", "feat", "outcome"]).proj_n.mean().unstack("outcome"))

    rng = np.random.default_rng(seed)
    stats = []
    for name, (plus, minus) in CONTRASTS.items():
        d = (unit_means[plus] - unit_means[minus]).dropna()
        D, p = stats_utils.sign_flip_test(d.to_numpy(), num_flips, rng)
        stats.append({"contrast": name, "plus": plus, "minus": minus, "D": D, "mean_d": d.mean(), "n_units": len(d), "p": p})
    stats = pd.DataFrame(stats)

    # Holm over the family: the smaller p doubled, the larger kept at least that, both capped at 1
    family = stats[stats.contrast.isin(HOLM_FAMILY)].sort_values("p")
    m = len(family)
    adjusted = np.minimum(1, np.maximum.accumulate(family.p.to_numpy() * (m - np.arange(m))))
    stats["p_holm"] = stats.contrast.map(dict(zip(family.contrast, adjusted)))

    cells = []
    for cell in TEST_CELLS:
        vals = unit_means[cell].dropna()
        cells.append({
            "cell": cell, "mean": vals.mean(), "se": vals.std(ddof=1) / np.sqrt(len(vals)),
            "n_units": len(vals), "n_events": int((events.cell == cell).sum()),
        })
    return stats, pd.DataFrame(cells)


def file_prefix(args):
    seed_str = "" if args.train_test_seed == 42 else f"_seed_{args.train_test_seed}"
    return f"{MODE}_{args.frs_source}{seed_str}"


def print_counts(events):
    per_session = events.groupby(["session", "cell"]).size().unstack("cell").fillna(0)
    print("\ntest events per session, pooled over features:", flush=True)
    print(pd.DataFrame({
        "median": per_session.median(), "min": per_session.min(), "sessions with 0": (per_session == 0).sum(),
    }).loc[TEST_CELLS].to_string(), flush=True)


def main(args):
    args.trial_interval = get_trial_interval(args.trial_event)
    args.time_range = PRE_STIM_RANGE
    subjects = ["SA", "BL"] if args.subject == "both" else [args.subject]

    events, axes = [], []
    for sub in subjects:
        sub_args = copy.deepcopy(args)
        sub_args.subject = sub
        feat_to_sessions, sub_sessions = get_feat_to_sessions(sub_args)
        for sess_name in sub_sessions.session_name:
            feats = [f for f in FEATURES if sess_name in feat_to_sessions[f]]
            print(f"{sub} session {sess_name}: {len(feats)} valid features", flush=True)
            sess_events, sess_axes = process_session(sess_name, feats, sub_args)
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
    parser.add_argument("--num_flips", default=10000, type=int)
    main(parser.parse_args())
