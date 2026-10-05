"""
Update projections onto pref and conf decoder axes, with the pair split, cell-mean projections and
sign-flip tests of mean_diff_update_projections.py. Replaces pref_conf_projection_updates.py, whose
decoders were trained on every trial they then scored, and whose inference ran on pseudo-trials
against a shuffle baseline.

The decoders come from decode_update_axes.py, trained on the pair split's axis trials only. For
each feature, mode and pre-stimulus bin, the axis is coef[high] - coef[low] averaged over the 8
decoder splits, applied to firing rates divided by the BatchNorm running std, as in
pref_conf_projection_updates.load_pref_vector. For each chose-X test trial k,

    proj = mean over bins of  sum over units of  (x[k+1] - x[k]) / std * weightsdiff

which is md.project_tests with the axis weightsdiff / std in firing-rate units. The axis norm, the
four contrasts and the sign flips over (session, feature) are md.compute_stats unchanged.

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
from scripts.pseudo_decoding.belief_partitions.pref_conf_projection_updates import load_pref_vector
from scripts.pseudo_decoding.belief_partitions.decode_update_axes import OUTPUT_PATH as DECODER_PATH
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


def load_axes(args, feats):
    """
    {(mode, feat): DataFrame of PseudoUnitID, TimeIdx, weightsdiff, std} and
    {(mode, feat): mean pre-stim decoder test accuracy}.
    """
    axes, accs = {}, {}
    for mode in md.AXES:
        for feat in feats:
            d_args = decoder_args(args, mode, feat)
            n_units = len(belief_partitions_io.read_units(copy.deepcopy(d_args), [feat]))
            n_coef = belief_partitions_io.read_models(copy.deepcopy(d_args), [feat]).models.iloc[0].coef_.shape[1]
            assert n_units == n_coef, f"{mode} {feat}: {n_units} unit ids but {n_coef} decoder inputs"
            weights = load_pref_vector(copy.deepcopy(d_args))
            axes[(mode, feat)] = weights[["PseudoUnitID", "TimeIdx", "weightsdiff", "std"]]
            dir_name = belief_partitions_io.get_dir_name(d_args, make_dir=False)
            accs[(mode, feat)] = np.load(os.path.join(dir_name, f"{feat}_{mode}_test_accs.npy")).mean()
    return axes, accs


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


def process_session(sess_name, feats, args, axes):
    """
    The test events' projections and the axis summaries for every valid feature of one session.
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
        trials = md.split_pairs(sess_name, feat, md.label_trials(beh, feat), args.train_test_seed)
        sess_axes, axis_row = {}, {"session": sess_name, "subject": args.subject, "feat": feat, "dropped": False}
        for mode in md.AXES:
            u, sq_norm, n = session_axis(axes[(mode, feat)], unit_ids, time_idxs)
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
        events_res.append(test[[
            "session", "subject", "feat", "TrialNumber", "outcome", "partition",
            "pref_cell", "conf_cell", "proj_pref", "proj_conf",
        ]])
    return events_res, axes_res


def main(args):
    args.trial_interval = get_trial_interval(args.trial_event)
    args.time_range = PRE_STIM_RANGE
    feats = args.feats.split(",") if args.feats else FEATURES
    axes, accs = load_axes(args, feats)

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
            sess_events, sess_axes = process_session(sess_name, sess_feats, sub_args, axes)
            events.extend(sess_events)
            axes_df.extend(sess_axes)
    events = pd.concat(events, ignore_index=True)
    axes_df = pd.DataFrame(axes_df)
    stats, cells = md.compute_stats(events, axes_df, args.num_flips, args.train_test_seed)

    out_args = copy.deepcopy(args)
    out_args.subject = "both"
    out_args.sig_unit_level = None
    output_dir = belief_partitions_io.get_dir_name(out_args)
    prefix = MODE if not args.feats else f"{MODE}_{'_'.join(feats)}"
    for name, df in [("events", events), ("axes", axes_df), ("stats", stats), ("cells", cells)]:
        df.to_pickle(os.path.join(output_dir, f"{prefix}_{name}.pickle"))

    print(f"\nsaved {prefix}_*.pickle to {output_dir}", flush=True)
    print("\nmean pre-stim decoder test accuracy:", flush=True)
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
