"""
The Belief state model's own trial-to-trial change in belief in feature X, summarized and tested as
decoder_update_projections.py summarizes the neural update projections, so the behavioral and
neural figures share their cells, conditions and statistics. Replaces the trial-level comparison
against all pooled changes in scripts/figure_generation/generate_change_in_beh_beliefs_by_obs_plots.py,
and uses both monkeys.

For each (session, feature X) of get_feat_to_sessions -- the sessions and features the neural run
uses -- and each valid trial k with a next valid trial, as in prior_dependent_updates.load_beh,

    db = b_{k+1}(X) - b_k(X),    b(X) = the model's {X}Prob

stored as proj_pref, so md.compute_stats and md.condition_summaries treat it as a pref-axis
projection, with (session, feature) as the unit and no normalization. Every trial is used: there
is no axis to hold trials out from. The cells and the pref contrasts use the chose-X trials; the
all-choice conditions also use the not-X trials.

--session_flips averages each contrast, and each cell's and condition's unit means for p_vs_0,
over features within session, and flips signs per session.

The model's beliefs are a deterministic function of behavior, so these tests describe the model's
updates rather than test a noisy measurement of them.

One process does both monkeys; no slurm launcher is needed.
"""

import os
import copy
import argparse
import numpy as np
import pandas as pd

from constants.behavioral_constants import *
from constants.decoding_constants import *
from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import BeliefPartitionConfigs, add_defaults_to_parser
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
from scripts.pseudo_decoding.belief_partitions.stim_belief_vector_alignment import get_feat_to_sessions
from scripts.pseudo_decoding.belief_partitions.prior_dependent_updates import load_beh
import scripts.pseudo_decoding.belief_partitions.mean_diff_update_projections as md

MODE = "belief_changes"
OUTPUT_PATH = "/data/patrick_res/belief_update_changes"
UNIT_COLS = ("session", "feat")
# db is stored as a pref-axis projection
AXIS_NAMES = ["pref"]


def session_changes(sess_name, feats, args):
    """
    One events frame per feature: every trial with a next valid trial, labeled by md.label_trials,
    with proj_pref = b_{k+1}(X) - b_k(X).
    """
    beh = load_beh(sess_name, args)
    res = []
    for feat in feats:
        trials = md.label_trials(beh, feat)
        prob = beh.set_index("TrialNumber")[f"{feat}Prob"]
        trials["proj_pref"] = trials.NextTrialNumber.map(prob) - trials.TrialNumber.map(prob)
        # the last trial has no next (NextTrialNumber -1), so no change
        trials = trials[trials.proj_pref.notna()].copy()
        trials["session"] = sess_name
        trials["subject"] = args.subject
        trials["feat"] = feat
        res.append(trials[[
            "session", "subject", "feat", "TrialNumber", "chose", "outcome", "partition",
            "pref_cell", "conf_cell", "proj_pref",
        ]])
    return res


def main(args):
    args.trial_interval = get_trial_interval(args.trial_event)
    events = []
    for sub in ["SA", "BL"]:
        sub_args = copy.deepcopy(args)
        sub_args.subject = sub
        feat_to_sessions, sub_sessions = get_feat_to_sessions(sub_args)
        for sess_name in sub_sessions.session_name:
            sess_feats = [f for f in FEATURES if sess_name in feat_to_sessions[f]]
            if not sess_feats:
                continue
            print(f"{sub} session {sess_name}: {len(sess_feats)} features", flush=True)
            events.extend(session_changes(sess_name, sess_feats, sub_args))
    events = pd.concat(events, ignore_index=True)

    stats, cells = md.compute_stats(events, None, args.num_flips, args.train_test_seed, unit_cols=UNIT_COLS,
                                    normalize=False, session_flips=args.session_flips, axis_names=AXIS_NAMES)
    conditions = md.condition_summaries(events, None, args.num_flips, args.train_test_seed, unit_cols=UNIT_COLS,
                                        normalize=False, session_flips=args.session_flips, axis_names=AXIS_NAMES)

    out_args = copy.deepcopy(args)
    out_args.subject = "both"
    out_args.sig_unit_level = None
    output_dir = belief_partitions_io.get_dir_name(out_args)
    prefix = MODE + ("_session_flips" if args.session_flips else "")
    for name, df in [("events", events), ("stats", stats), ("cells", cells), ("conditions", conditions)]:
        df.to_pickle(os.path.join(output_dir, f"{prefix}_{name}.pickle"))

    print(f"\nsaved {prefix}_*.pickle to {output_dir}", flush=True)
    print(events.groupby("subject").agg(sessions=("session", "nunique"), events=("TrialNumber", "size")).to_string(), flush=True)
    print("\ncells:", flush=True)
    print(cells.to_string(), flush=True)
    print("\nconditions:", flush=True)
    print(conditions.to_string(), flush=True)
    print("\nstats:", flush=True)
    print(stats.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(
        BeliefPartitionConfigs(mode=MODE, subject="both", base_output_path=OUTPUT_PATH), parser
    )
    parser.add_argument("--num_flips", default=10000, type=int)
    # sign flips per session, each contrast and unit mean averaged over features first
    parser.add_argument("--session_flips", action="store_true")
    main(parser.parse_args())
