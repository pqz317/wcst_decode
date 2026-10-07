"""
Update projections onto pref and conf decoder axes, with the pair split, cell-mean projections and
sign-flip tests of mean_diff_update_projections.py. Replaces pref_conf_projection_updates.py, whose
decoders were trained on every trial they then scored, and whose inference ran on pseudo-trials
against a shuffle baseline.

The decoders come from decode_update_axes.py, one per pair split, each trained on its split's
axis trials only. For each feature, mode, split and pre-stimulus bin, the axis is
coef[high] - coef[low] of that split's decoder, applied to firing rates divided by its BatchNorm
running std, as in pref_conf_projection_updates.load_pref_vector but without averaging over
splits. For each chose-X test trial k of split i, on split i's axis,

    proj = mean over bins of  sum over units of  (x[k+1] - x[k]) / std * weightsdiff

which is md.project_tests with the axis weightsdiff / std in firing-rate units. The four contrasts
and the sign flips are md.compute_stats with (session, feature, split) as the unit.

Each cell and each condition below also gets p_vs_0, a two-sided sign-flip test of its unit means
against 0.

--session_flips instead averages each contrast, and each cell's and condition's unit means for
p_vs_0, over splits, then over features, within session, and flips signs per session.

--normalize_axes divides each projection by its axis norm, sqrt(sum of weightsdiff^2 over every
unit and bin of that (feature, split)'s decoder), so each decoder's axis has unit length. Off by
default: projections are then in the decoder's own weight scale.

The conditions of pref_conf_projection_updates.py are also summarized, each as the mean and SE
across (session, feature, split) units of the unit's mean projection over its test trials:

    chose X / cor, chose X / inc:  every chose-X test trial of that outcome, over partitions
    cor, inc:                      with --split_method chunk only, also the not-X trials of the test
                                   blocks whose next trial is in the same block. The decoders never
                                   trained on them. Under the pair split every not-X trial is an
                                   axis trial, so these two are left out.

The not-X trials are in the saved events, with pref_cell and conf_cell "not X", so the four
contrasts and the per-cell summaries still use only the chose-X test trials.

--old_axes projects onto the old no-cond decoders of decode_belief_partitions.py instead, which
were trained on every trial, test trials included. Their 8 models are averaged into one axis per
bin, as pref_conf_projection_updates.load_pref_vector does, and that axis is used for all 8 pair
splits. Everything else -- test trials, projection, statistics -- is unchanged, so the comparison
with the default run isolates the training-trial overlap of the old axes.

--per_region runs the whole analysis once per region of REGIONS_OF_INTEREST, on the decoders
decode_update_axes.py trained on that region's units, projecting each session's units in that
region only. Each region's results go to its own directory, named by get_dir_name with the region.
(session, feature) units with no decoder units in the region have no projection and drop out of
the stats.

Runs locally after every decode_update_axes.py job has finished.
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
from scripts.pseudo_decoding.belief_partitions.stim_belief_vector_alignment import get_feat_to_sessions, load_session_frs
from scripts.pseudo_decoding.belief_partitions.prior_dependent_updates import load_beh, PRE_STIM_RANGE
from scripts.pseudo_decoding.belief_partitions.pref_conf_projection_updates import MODE_TO_DIRECTION_LABELS
from scripts.pseudo_decoding.belief_partitions.decode_update_axes import (
    pair_splits, args_base_path, add_split_args, PRE_STIM_BINS, DEFAULT_TEST_FRAC,
)
import scripts.pseudo_decoding.belief_partitions.mean_diff_update_projections as md

MODE = "decoder_updates"
# the old no-cond decoders, trained on every trial
OLD_DECODER_PATH = "/data/patrick_res/belief_partitions"
OUTPUT_PATH = "/data/patrick_res/decoder_update_projections"
SIG_UNIT_LEVEL = "{mode}_99th_no_cond_window_filter_drift"
# BatchNorm std below this (firing-rate units) drops a (unit, bin) axis entry. 0.25 removes the
# entries at or near the numerical floor (sqrt(1e-5) ~ 0.003): ~6% of pref and ~1% of conf
# entries. In a sweep of 0.1, 0.25, 0.5 and 1, every cell mean was stable from 0.25 to 0.5, while
# at 0.1 the largest inc/High X (session, feature, split) mean was still 7.4, against 3.3 at 0.25
DEFAULT_MIN_STD = 0.25
REGION_LEVEL = "structure_level2_cleaned"


def decoder_args(args, mode, feat):
    """
    args locating the decoder run for (mode, feat): decode_update_axes.py's, or with --old_axes the
    old no-cond run of decode_belief_partitions.py.
    """
    d_args = copy.deepcopy(args)
    d_args.subject = "both"
    d_args.mode = mode
    d_args.feat = feat
    d_args.sig_unit_level = SIG_UNIT_LEVEL.format(mode=mode)
    d_args.base_output_path = OLD_DECODER_PATH if args.old_axes else args_base_path(args)
    d_args.shuffle_idx = None
    return d_args


def split_weights(d_args, mode, feat):
    """
    One split's axis per row: DataFrame of split, PseudoUnitID, TimeIdx, weightsdiff, std. The same
    quantities and TimeIdx convention as pref_conf_projection_updates.load_pref_vector, kept per
    split instead of averaged.
    """
    high_idx = MODE_TO_CLASSES[mode].index(MODE_TO_DIRECTION_LABELS[mode]["high"])
    low_idx = MODE_TO_CLASSES[mode].index(MODE_TO_DIRECTION_LABELS[mode]["low"])
    models = belief_partitions_io.read_models(copy.deepcopy(d_args), [feat])
    units = belief_partitions_io.read_units(copy.deepcopy(d_args), [feat]).PseudoUnitID.to_numpy()
    res = []
    for row in models.itertuples():
        coef = row.models.coef_
        assert coef.shape[1] == len(units), f"{mode} {feat}: {len(units)} unit ids but {coef.shape[1]} decoder inputs"
        # 1e-5 from torch batchnorm1d, numerical
        std = np.sqrt(row.models.model.norm.running_var.detach().cpu().numpy() + 1e-5)
        res.append(pd.DataFrame({
            "split": int(row.run), "PseudoUnitID": units,
            "TimeIdx": int(round((row.Time - 0.1) * 10)),
            "weightsdiff": coef[high_idx, :] - coef[low_idx, :], "std": std,
        }))
    return pd.concat(res, ignore_index=True)


def averaged_weights(weights, num_splits):
    """
    The old analysis's axis: weightsdiff and std averaged over the decoder's models per
    (unit, bin), as in load_pref_vector, repeated once per pair split.
    """
    avg = weights.groupby(["PseudoUnitID", "TimeIdx"])[["weightsdiff", "std"]].mean().reset_index()
    return pd.concat([avg.assign(split=i) for i in range(num_splits)], ignore_index=True)


def load_axes(args, feats):
    """
    {(mode, feat): split_weights}, {(mode, feat): mean pre-stim decoder test accuracy over splits},
    and {(mode, feat): the splits table the decoders were trained with}, None with --old_axes,
    whose decoders were not trained on pair splits.
    """
    axes, accs, splits = {}, {}, {}
    for mode in md.AXES:
        for feat in feats:
            d_args = decoder_args(args, mode, feat)
            weights = split_weights(d_args, mode, feat)
            axes[(mode, feat)] = averaged_weights(weights, args.num_splits) if args.old_axes else weights
            dir_name = belief_partitions_io.get_dir_name(d_args, make_dir=False)
            # the old decoders cover the whole StimOnset interval; its first 10 bins are pre-stim
            accs[(mode, feat)] = np.load(os.path.join(dir_name, f"{feat}_{mode}_test_accs.npy"))[:len(PRE_STIM_BINS)].mean()
            splits[(mode, feat)] = None if args.old_axes else pd.read_pickle(os.path.join(dir_name, f"{feat}_{mode}_splits.pickle"))
    return axes, accs, splits


def check_split(trials, table, sess_name, split_idx):
    """
    Asserts the decoder for this split trained only on axis trials and was tested only on test
    trials of the split rebuilt here.
    """
    rows = table[(table.session == sess_name) & (table.split_idx == split_idx)]
    half = trials.set_index("TrialNumber").half
    train = np.concatenate(rows.TrainTrials.values)
    test = np.concatenate(rows.TestTrials.values)
    assert (half.loc[train] == "axis").all(), f"session {sess_name} split {split_idx}: train trial not an axis trial"
    assert (half.loc[test] == "test").all(), f"session {sess_name} split {split_idx}: test trial not a test trial"


def session_axis(weights, unit_ids, time_idxs, min_std=0.0):
    """
    The axis weightsdiff / std as a units x bins array over this session's units, 0 for units not in
    the decoder, plus its squared norm sum(weightsdiff^2) over the session's decoder units, the
    number of decoder units, and the number of (unit, bin) entries dropped by min_std.

    Entries whose BatchNorm std is below min_std are set to 0, in the projection and the norm.
    Such a unit was nearly silent in that bin on the decoder's training trials, so its weight is
    barely trained, and dividing a test trial's change by that std multiplies it by up to ~300.
    """
    w = weights[weights.PseudoUnitID.isin(unit_ids)]
    if len(w) == 0:
        return np.zeros((len(unit_ids), len(time_idxs))), 0.0, 0, 0
    grid = w.pivot(index="PseudoUnitID", columns="TimeIdx", values="weightsdiff").reindex(index=unit_ids, columns=time_idxs)
    std = w.pivot(index="PseudoUnitID", columns="TimeIdx", values="std").reindex(index=unit_ids, columns=time_idxs)
    n_units = int(grid.notna().any(axis=1).sum())
    low = std < min_std
    grid = grid.mask(low)
    u = (grid / std).fillna(0).to_numpy()
    return u, float((grid.fillna(0).to_numpy() ** 2).sum()), n_units, int(low.to_numpy().sum())


def process_session(sess_name, feats, args, axes, tables):
    """
    The test events' projections and the axis summaries for every valid feature and split of one
    session, each split's test trials projected on that split's axes.
    """
    beh = load_beh(sess_name, args)
    frs_args = copy.deepcopy(args)
    frs_args.sig_unit_level = None
    session_frs = load_session_frs(sess_name, frs_args)
    if session_frs is None:
        return [], []
    X, rows, unit_ids = session_frs
    # bins in the order load_session_frs stacks them, as TimeIdx: -10 .. -1 for the pre-stim window
    time_idxs = list(range(int(PRE_STIM_RANGE[0] / 100), int(PRE_STIM_RANGE[1] / 100)))
    assert X.shape[2] == len(time_idxs)

    events_res, axes_res = [], []
    for feat in feats:
        for split_idx, trials in pair_splits(sess_name, feat, args).items():
            sess_axes = {}
            axis_row = {"session": sess_name, "subject": args.subject, "feat": feat, "split": split_idx, "dropped": False}
            for mode in md.AXES:
                if tables[(mode, feat)] is not None and sess_name in set(tables[(mode, feat)].session):
                    check_split(trials, tables[(mode, feat)], sess_name, split_idx)
                weights = axes[(mode, feat)]
                u, sq_norm, n, n_low = session_axis(weights[weights.split == split_idx], unit_ids, time_idxs, args.min_std)
                sess_axes[mode] = u
                axis_row[f"sq_norm_{mode}"] = sq_norm
                axis_row[f"n_units_{mode}"] = n
                axis_row[f"n_low_std_{mode}"] = n_low
            axes_res.append(axis_row)

            include = (trials.half == "test") & trials.chose
            if args.split_method == "chunk":
                # the not-X trials of in-block pairs, which the decoders never trained on
                include |= trials.in_block_pair & ~trials.chose
            test = md.project_tests(X, rows, trials, sess_axes, include=include)
            for mode in md.AXES:
                if axis_row[f"n_units_{mode}"] == 0:
                    test[f"proj_{mode}"] = np.nan
            test["session"] = sess_name
            test["subject"] = args.subject
            test["feat"] = feat
            test["split"] = split_idx
            events_res.append(test[[
                "session", "subject", "feat", "split", "TrialNumber", "chose", "outcome", "partition",
                "pref_cell", "conf_cell", "proj_pref", "proj_conf",
            ]])
    return events_res, axes_res


def run(args):
    """
    The whole analysis for one population: all units, or with args.regions one region's.
    """
    feats = args.feats.split(",") if args.feats else FEATURES
    axes, accs, tables = load_axes(args, feats)

    events, axes_df = [], []
    for sub in ["SA", "BL"]:
        sub_args = copy.deepcopy(args)
        sub_args.subject = sub
        feat_to_sessions, sub_sessions = get_feat_to_sessions(sub_args)
        for sess_name in sub_sessions.session_name:
            sess_feats = [f for f in feats if sess_name in feat_to_sessions[f]]
            if not sess_feats:
                continue
            print(f"{sub} session {sess_name}: {len(sess_feats)} features", flush=True)
            sess_events, sess_axes = process_session(sess_name, sess_feats, sub_args, axes, tables)
            events.extend(sess_events)
            axes_df.extend(sess_axes)
    events = pd.concat(events, ignore_index=True)
    axes_df = pd.DataFrame(axes_df)
    stats, cells = md.compute_stats(events, axes_df, args.num_flips, args.train_test_seed,
                                    unit_cols=("session", "feat", "split"), normalize=args.normalize_axes,
                                    session_flips=args.session_flips)
    # the all-choice conditions need the chunk split's not-X test-block trials
    conditions = md.condition_summaries(events, axes_df, args.num_flips, args.train_test_seed,
                                        unit_cols=("session", "feat", "split"), normalize=args.normalize_axes,
                                        session_flips=args.session_flips,
                                        include_all_choice=args.split_method == "chunk")

    out_args = copy.deepcopy(args)
    out_args.subject = "both"
    out_args.sig_unit_level = None
    output_dir = belief_partitions_io.get_dir_name(out_args)
    prefix = MODE if not args.feats else f"{MODE}_{'_'.join(feats)}"
    if args.normalize_axes:
        prefix += "_axis_norm"
    if args.old_axes:
        prefix += "_old_axes"
    if args.split_method == "chunk":
        prefix += f"_chunk_{args.chunk_test_len}_train_frac_{args.chunk_train_frac:g}"
    elif not np.isclose(args.test_frac, DEFAULT_TEST_FRAC):
        prefix += f"_test_frac_{args.test_frac:g}"
    if not np.isclose(args.min_std, DEFAULT_MIN_STD):
        prefix += f"_min_std_{args.min_std:g}"
    if args.session_flips:
        prefix += "_session_flips"
    for name, df in [("events", events), ("axes", axes_df), ("stats", stats), ("cells", cells), ("conditions", conditions)]:
        df.to_pickle(os.path.join(output_dir, f"{prefix}_{name}.pickle"))

    print(f"\nsaved {prefix}_*.pickle to {output_dir}", flush=True)
    print("\nmean pre-stim decoder test accuracy, over splits:", flush=True)
    print(pd.Series(accs).unstack(0).round(3).to_string(), flush=True)
    print("\ncells:", flush=True)
    print(cells.to_string(), flush=True)
    print("\nconditions:", flush=True)
    print(conditions.to_string(), flush=True)
    print("\nstats:", flush=True)
    print(stats.to_string(), flush=True)


def main(args):
    args.trial_interval = get_trial_interval(args.trial_event)
    args.time_range = PRE_STIM_RANGE
    if not args.per_region:
        run(args)
        return
    for region in REGIONS_OF_INTEREST:
        print(f"\n===== region {region} =====", flush=True)
        region_args = copy.deepcopy(args)
        region_args.region_level = REGION_LEVEL
        region_args.regions = region
        run(region_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(
        BeliefPartitionConfigs(mode=MODE, subject="both", base_output_path=OUTPUT_PATH), parser
    )
    parser.add_argument("--num_flips", default=10000, type=int)
    # comma separated feature names, for checking the pipeline on a subset; default all 12
    parser.add_argument("--feats", default=None, type=str)
    parser.add_argument("--normalize_axes", action="store_true")
    # sign flips per session, each contrast averaged over splits then features first
    parser.add_argument("--session_flips", action="store_true")
    parser.add_argument("--old_axes", action="store_true")
    # must match the decode_update_axes.py run: it sets which decoders are read and rebuilds the splits
    parser.add_argument("--test_frac", default=DEFAULT_TEST_FRAC, type=float)
    add_split_args(parser)
    # (unit, bin) axis entries with BatchNorm std below this, in firing-rate units, are left out;
    # 0 keeps every entry
    parser.add_argument("--min_std", default=DEFAULT_MIN_STD, type=float)
    # runs once per region of interest, on that region's decoders, instead of the whole population
    parser.add_argument("--per_region", action="store_true")
    main(parser.parse_args())
