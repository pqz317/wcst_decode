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

which is md.project_tests with the axis weightsdiff / std in firing-rate units. The axis norm (per
feature and split), the four contrasts and the sign flips are md.compute_stats with
(session, feature, split) as the unit.

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
from scripts.pseudo_decoding.belief_partitions.decode_update_axes import OUTPUT_PATH as DECODER_PATH, pair_splits
import scripts.pseudo_decoding.belief_partitions.mean_diff_update_projections as md

MODE = "decoder_updates"
OUTPUT_PATH = "/data/patrick_res/decoder_update_projections"
SIG_UNIT_LEVEL = "{mode}_99th_no_cond_window_filter_drift"


def decoder_args(args, mode, feat):
    """
    args locating the decode_update_axes.py run for (mode, feat).
    """
    d_args = copy.deepcopy(args)
    d_args.subject = "both"
    d_args.mode = mode
    d_args.feat = feat
    d_args.sig_unit_level = SIG_UNIT_LEVEL.format(mode=mode)
    d_args.base_output_path = DECODER_PATH
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


def load_axes(args, feats):
    """
    {(mode, feat): split_weights}, {(mode, feat): mean pre-stim decoder test accuracy over splits},
    and {(mode, feat): the splits table the decoders were trained with}.
    """
    axes, accs, splits = {}, {}, {}
    for mode in md.AXES:
        for feat in feats:
            d_args = decoder_args(args, mode, feat)
            axes[(mode, feat)] = split_weights(d_args, mode, feat)
            dir_name = belief_partitions_io.get_dir_name(d_args, make_dir=False)
            accs[(mode, feat)] = np.load(os.path.join(dir_name, f"{feat}_{mode}_test_accs.npy")).mean()
            splits[(mode, feat)] = pd.read_pickle(os.path.join(dir_name, f"{feat}_{mode}_splits.pickle"))
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


def session_axis(weights, unit_ids, time_idxs):
    """
    The axis weightsdiff / std as a units x bins array over this session's units, 0 for units not in
    the decoder, plus its squared norm sum(weightsdiff^2) over the session's decoder units.
    """
    w = weights[weights.PseudoUnitID.isin(unit_ids)]
    if len(w) == 0:
        return np.zeros((len(unit_ids), len(time_idxs))), 0.0, 0
    grid = w.pivot(index="PseudoUnitID", columns="TimeIdx", values="weightsdiff").reindex(index=unit_ids, columns=time_idxs)
    std = w.pivot(index="PseudoUnitID", columns="TimeIdx", values="std").reindex(index=unit_ids, columns=time_idxs)
    u = (grid / std).fillna(0).to_numpy()
    return u, float((grid.fillna(0).to_numpy() ** 2).sum()), int(grid.notna().any(axis=1).sum())


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
                if sess_name in set(tables[(mode, feat)].session):
                    check_split(trials, tables[(mode, feat)], sess_name, split_idx)
                weights = axes[(mode, feat)]
                u, sq_norm, n = session_axis(weights[weights.split == split_idx], unit_ids, time_idxs)
                sess_axes[mode] = u
                axis_row[f"sq_norm_{mode}"] = sq_norm
                axis_row[f"n_units_{mode}"] = n
            axes_res.append(axis_row)

            test = md.project_tests(X, rows, trials, sess_axes)
            for mode in md.AXES:
                if axis_row[f"n_units_{mode}"] == 0:
                    test[f"proj_{mode}"] = np.nan
            test["session"] = sess_name
            test["subject"] = args.subject
            test["feat"] = feat
            test["split"] = split_idx
            events_res.append(test[[
                "session", "subject", "feat", "split", "TrialNumber", "outcome", "partition",
                "pref_cell", "conf_cell", "proj_pref", "proj_conf",
            ]])
    return events_res, axes_res


def main(args):
    args.trial_interval = get_trial_interval(args.trial_event)
    args.time_range = PRE_STIM_RANGE
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
                                    unit_cols=("session", "feat", "split"))

    out_args = copy.deepcopy(args)
    out_args.subject = "both"
    out_args.sig_unit_level = None
    output_dir = belief_partitions_io.get_dir_name(out_args)
    prefix = MODE if not args.feats else f"{MODE}_{'_'.join(feats)}"
    for name, df in [("events", events), ("axes", axes_df), ("stats", stats), ("cells", cells)]:
        df.to_pickle(os.path.join(output_dir, f"{prefix}_{name}.pickle"))

    print(f"\nsaved {prefix}_*.pickle to {output_dir}", flush=True)
    print("\nmean pre-stim decoder test accuracy, over splits:", flush=True)
    print(pd.Series(accs).unstack(0).round(3).to_string(), flush=True)
    print("\ncells:", flush=True)
    print(cells.to_string(), flush=True)
    print("\nstats:", flush=True)
    print(stats.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(
        BeliefPartitionConfigs(mode=MODE, subject="both", base_output_path=OUTPUT_PATH), parser
    )
    parser.add_argument("--num_flips", default=10000, type=int)
    # comma separated feature names, for checking the pipeline on a subset; default all 12
    parser.add_argument("--feats", default=None, type=str)
    main(parser.parse_args())
