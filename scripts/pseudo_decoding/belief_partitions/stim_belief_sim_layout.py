"""
Step 1 of the stim/belief projection simulation (claude_notes/Stim Belief projection simulation
proposal.md): the layout of the real runs, and the gaps the simulation is calibrated to.

The simulation replaces firing rates with Gaussian draws but keeps everything else about the real
runs, so it needs, per (population, feature, split variant):

- the sessions, and each session's number of units in the A-vs-B stim axis run (all units)
- each session's number of pref units, pref_99th_window_filter_drift, the units the B-vs-C
  preference run and the projection use
- each session's real trial counts per condition, post-balance, in both runs

All of it is read off the runs already on disk: trial counts from the `_splits.pickle`s (split 0,
TrialNumbers are the post-balance trials of each condition, before the train/test split), units from
the `_unit_ids.csv`s (PseudoUnitID // 100 is the session, see SessionData.generate_pseudo_data).

Split variants:

- "b_split": axis run fit on B1 (`b_split_half=1`), pref run on B2 (`b_split_half=2`). The two
  draw from disjoint halves of B.
- "no_split": the earlier leaky runs. Both use all of B, so the simulation draws a single B pool per
  session, of the axis run's Chose count (the axis run balances A down to B and A never binds, so
  that is all of B), and the pref run's B is a subsample of it.

Targets, per population, feedback aligned:

- stim gap: true - shuffle of the axis run, averaged over PEAK_WINDOW, -0.6 to -0.2s before
  feedback. That is where stim decoding peaks in the whole population, ITC and AMY, while the choice
  is being made; after feedback it has already fallen off. One fixed window for every population
  rather than each one's own top bins, which in DSTR, LPFC and ACC are scattered noise.
- pref gap: true - shuffle of the pref run over the same window, so both targets describe the same
  moment. Also over all bins, for reference.
- proj gap: the observed projection's true - shuffle, over the window and over all bins, to mark on
  the figures.

    python3 scripts/pseudo_decoding/belief_partitions/stim_belief_sim_layout.py

Writes LAYOUT_PATH, a dict {"layouts": {(population, feat, split): df}, "targets": df}.
"""

import os
import argparse
import numpy as np
import pandas as pd

from constants.behavioral_constants import *
from constants.decoding_constants import *
from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import BeliefPartitionConfigs
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
from scripts.pseudo_decoding.belief_partitions.decode_pref_on_choice_axis import get_proj_mode, PROJ_OUTPUT_PATH, CHOICE_AXIS_PATH

OUTPUT_PATH = "/data/patrick_res/stim_belief_projection_sim"
LAYOUT_PATH = os.path.join(OUTPUT_PATH, "layout.pickle")

TRIAL_EVENT = "FeedbackOnsetLong"
PREF_PATH = "/data/patrick_res/belief_partitions"
AXIS_BEH_FILTERS = {"Response": "Correct", "BeliefPartition": "High Not X"}
PREF_BEH_FILTERS = {"Response": "Correct", "Choice": "Chose"}
PREF_SIG_UNIT_LEVEL = "pref_99th_window_filter_drift"
REGION_LEVEL = "structure_level2_cleaned"

# "whole_pop" is a sentinel for the no-region-filter run, same as the launchers
POPULATIONS = ["whole_pop"] + REGIONS_OF_INTEREST

# (axis run's b_split_half, pref run's b_split_half) per split variant
SPLIT_VARIANTS = {"b_split": (1, 2), "no_split": (None, None)}
# the variant whose stim and pref gaps are the calibration targets. Its layout is the one the
# calibration's noise power comes from, and no_split reuses the resulting s, p
TARGETS_SPLIT = "b_split"

# bin times, inclusive, in seconds relative to feedback
PEAK_WINDOW = (-0.6, -0.2)


def get_args(population, mode, b_split_half):
    """
    Configs of one real run: the A-vs-B stim axis run for mode "choice", else the B-vs-C preference
    run, with its projection stored under the preference run's name
    """
    args = argparse.Namespace(**BeliefPartitionConfigs()._asdict())
    args.subject = "both"
    args.trial_event = TRIAL_EVENT
    args.trial_interval = get_trial_interval(TRIAL_EVENT)
    args.mode = mode
    args.b_split_half = b_split_half
    if population == "whole_pop":
        args.region_level, args.regions = None, None
    else:
        args.region_level, args.regions = REGION_LEVEL, population
    if mode == "choice":
        args.beh_filters = AXIS_BEH_FILTERS
        args.sig_unit_level = None
        args.base_output_path = CHOICE_AXIS_PATH
    else:
        args.beh_filters = PREF_BEH_FILTERS
        args.sig_unit_level = PREF_SIG_UNIT_LEVEL
        args.base_output_path = PREF_PATH
    return args


def read_counts(args, feat):
    """
    Per session trial counts of each condition, post-balance, as session x condition
    """
    args.feat = feat
    args.shuffle_idx = None
    run_dir = belief_partitions_io.get_dir_name(args, make_dir=False)
    splits = pd.read_pickle(os.path.join(run_dir, f"{belief_partitions_io.get_file_name(args)}_splits.pickle"))
    splits = splits[splits.split_idx == 0]
    counts = splits.assign(n=splits.TrialNumbers.apply(len)).pivot(index="session", columns="Condition", values="n")
    counts.index = counts.index.astype(int)
    return counts


def read_units_per_session(args, feat):
    units = belief_partitions_io.read_units(args, [feat])
    return (units.PseudoUnitID // 100).value_counts()


def layout_for(population, feat, split):
    """
    One row per session in the axis run: its units, its pref units, and its trial counts. Sessions
    absent from the pref run (no pref units, or no trials) have zero pref units and pref counts
    """
    axis_half, pref_half = SPLIT_VARIANTS[split]
    axis_args = get_args(population, "choice", axis_half)
    pref_args = get_args(population, "pref", pref_half)

    axis_counts = read_counts(axis_args, feat)
    pref_counts = read_counts(pref_args, feat)
    axis_units = read_units_per_session(axis_args, feat)
    pref_units = read_units_per_session(pref_args, feat)

    layout = pd.DataFrame({
        "session": axis_counts.index,
        "n_units": axis_units.reindex(axis_counts.index).to_numpy(),
        "n_pref_units": pref_units.reindex(axis_counts.index).fillna(0).astype(int).to_numpy(),
        "n_A": axis_counts["Not Chose"].to_numpy(),
        "n_B_axis": axis_counts["Chose"].to_numpy(),
        "n_B_pref": pref_counts["High Not X"].reindex(axis_counts.index).fillna(0).astype(int).to_numpy(),
        "n_C": pref_counts["High X"].reindex(axis_counts.index).fillna(0).astype(int).to_numpy(),
    })
    missing = set(pref_counts.index) - set(axis_counts.index)
    if missing:
        raise ValueError(f"{population} {feat} {split}: pref sessions {missing} missing from the axis run")
    if layout.n_units.isna().any():
        raise ValueError(f"{population} {feat} {split}: sessions with trials but no units")
    if (layout.n_pref_units > layout.n_units).any():
        raise ValueError(f"{population} {feat} {split}: more pref units than units in a session")
    # a session drops out of the pref run when it has no pref units, so the two should agree
    if ((layout.n_pref_units > 0) != (layout.n_C > 0)).any():
        raise ValueError(f"{population} {feat} {split}: pref units and pref trials disagree on sessions")
    return layout


def gap(res, mode):
    """
    true - shuffle, averaged over features and runs, per time bin
    """
    true = res[res["mode"] == mode].groupby("Time").Accuracy.mean()
    shuffle = res[res["mode"] == f"{mode}_shuffle"].groupby("Time").Accuracy.mean()
    return true - shuffle


def in_window(times):
    times = np.asarray(times)
    return (times > PEAK_WINDOW[0] - 1e-6) & (times < PEAK_WINDOW[1] + 1e-6)


def targets_for(population):
    """
    Calibration targets and the observed projection gap, for the B split runs, plus the observed
    projection gap for the no split runs
    """
    row = {"population": population}

    axis_args = get_args(population, "choice", 1)
    stim_res = belief_partitions_io.read_results(axis_args, FEATURES)
    stim_gap = gap(stim_res, "choice")
    row["stim_window"] = np.round(stim_gap.index[in_window(stim_gap.index)], 1)
    row["stim_gap_peak"] = stim_gap[in_window(stim_gap.index)].mean()
    row["stim_gap_all"] = stim_gap.mean()

    pref_args = get_args(population, "pref", 2)
    pref_gap = gap(belief_partitions_io.read_results(pref_args, FEATURES), "pref")
    row["pref_gap_peak"] = pref_gap[in_window(pref_gap.index)].mean()
    row["pref_gap_all"] = pref_gap.mean()

    for split, (axis_half, pref_half) in SPLIT_VARIANTS.items():
        proj_args = get_args(population, "pref", pref_half)
        proj_args.mode = get_proj_mode(AXIS_BEH_FILTERS, axis_half)
        proj_args.base_output_path = PROJ_OUTPUT_PATH
        proj_gap = gap(belief_partitions_io.read_results(proj_args, FEATURES), proj_args.mode)
        row[f"proj_gap_all_{split}"] = proj_gap.mean()
        row[f"proj_gap_peak_{split}"] = proj_gap[in_window(proj_gap.index)].mean()
    return row


def main():
    layouts = {}
    for population in POPULATIONS:
        for feat in FEATURES:
            for split in SPLIT_VARIANTS:
                layouts[(population, feat, split)] = layout_for(population, feat, split)
        print(f"read layouts for {population}", flush=True)
    targets = pd.DataFrame([targets_for(population) for population in POPULATIONS])
    print(targets.drop(columns="stim_window").round(3).to_string())

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    pd.to_pickle({"layouts": layouts, "targets": targets}, LAYOUT_PATH)
    print(f"wrote {LAYOUT_PATH}")


if __name__ == "__main__":
    main()
