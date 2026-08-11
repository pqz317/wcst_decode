"""
Script for measuring how aligned the choice population vector is with the preference population
vector, a time bin at a time, on the whole population.

For each feature and time bin, builds two mean-difference vectors over z-scored activity,

    v_choice = mean(r | Chose X)  - mean(r | Not Chose X)
    v_pref   = mean(r | High X)   - mean(r | High Not X)

with per-unit coordinates concatenated across sessions, and reports cos(v_choice, v_pref).

This is the encoding counterpart to decode_pref_on_choice_axis.py, which asks the readout
question with decoder weights. See claude_notes/weight_vs_mean_difference_axes.md for why the
two are different quantities rather than two estimators of the same one, and section 7
recommendation 1 there for why this version is worth having: each unit's coordinate is computed
independently of every other unit, so no decoder runs are needed and unit sets can be
intersected freely.

Which trials the choice vector is built from is set by --choice_beh_filters: {} for all valid
trials, {"Response": "Correct"} to restrict it to correct trials, which matches the pool v_pref is
built from. The filters are part of the mode results are stored under, so runs against differently
filtered choice vectors sit side by side. This is the encoding counterpart of
decode_pref_on_choice_axis.py's --axis_beh_filters, which picks the same restriction on the
decoder side by choosing which choice run's axis to read.

IMPORTANT interpretation limit: cos_sim is NOT deattenuated. Finite-trial noise inflates both
norms in the denominator, by a factor that differs across time bins, so the number is only
meaningful as a contrast against its shuffle. See claude_notes/cosine_similarity_debiasing.md
for the measured attenuation and for the two recipes that would fix it.

One job per (trial_event, shuffle) is what slurm_launch_choice_pref_alignment.sh submits; passing
--run_all instead walks that whole grid in this process, writing exactly the same files. That is
strictly cheaper than the 22 jobs: a session's firing rates don't depend on the shuffle, so with
sessions as the outer loop each pickle is read once per event and serves every shuffle, rather
than once per (event, shuffle).
"""

import os
import numpy as np
import pandas as pd

import utils.behavioral_utils as behavioral_utils
import utils.spike_utils as spike_utils
import utils.classifier_utils as classifier_utils

from constants.behavioral_constants import *
from constants.decoding_constants import *

import argparse
import copy
import functools
import json
from distutils.util import strtobool
from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import BeliefPartitionConfigs, add_defaults_to_parser
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
import scripts.pseudo_decoding.belief_partitions.decode_belief_partitions as decode_belief_partitions

# mode results are stored under
MODE = "choice_pref_align"
OUTPUT_PATH = "/data/patrick_res/choice_pref_alignment"

# the two contrasts being compared: the mode supplying the `condition` column, and its filters.
# these mirror the existing runs -- pref is the fig 5 "preference | selected" config, choice is
# the all-units choice config behind /data/patrick_res/choice_reward. The choice filters are the
# default only: --choice_beh_filters replaces them, see get_contrasts
CONTRASTS = {
    "choice": {"mode": "choice", "beh_filters": {}},
    "pref": {"mode": "pref", "beh_filters": {"Response": "Correct", "Choice": "Chose"}},
}

# a condition with fewer trials than this can't give a usable mean. With the per-feature session
# restriction in force this should almost never fire -- the drop count is logged so it's visible
MIN_TRIALS_PER_COND = 2

# populations the cosine is reported for. Every unit's coordinate is computed independently of
# every other unit's, so a region is a subset of the same per-unit vectors rather than a separate
# run -- no region loop over sessions, no refitting, one pass serves all of these
WHOLE_POP = "whole_pop"

# a cosine over a handful of units is meaningless; regions below this for a (feat, bin) are skipped
MIN_UNITS_PER_REGION = 5

REGION_LEVEL = "structure_level2_cleaned"

# balance conditions the way the decoding runs do, so the trials are identical to theirs.
# a mean difference doesn't need it, so flipping this off is the natural robustness pass
BALANCE_CONDITIONS = True

DATA_MODE = "FiringRate"

# the grid --run_all covers, mirroring slurm_launch_choice_pref_alignment.sh: both events, the
# true run plus 10 shuffles. Overridable with --trial_events / --num_shuffles
ALL_TRIAL_EVENTS = ["StimOnset", "FeedbackOnsetLong"]
NUM_SHUFFLES = 10


def get_contrasts(choice_beh_filters):
    """
    The two contrasts to compute, with the choice vector's trials taken from --choice_beh_filters.

    Only the choice side is configurable: the preference contrast is the fig 5 config the whole
    analysis is anchored to, and its filters are what make the two vectors' trial pools nest
    (every pref trial is a Chose X trial), which is what the docstring's cancellation argument
    rests on.
    """
    contrasts = copy.deepcopy(CONTRASTS)
    contrasts["choice"]["beh_filters"] = choice_beh_filters
    return contrasts


def get_mode(choice_beh_filters):
    """
    Mode results are stored under. The choice contrast's filters are part of the name, so runs
    against different choice vectors sit side by side rather than overwriting each other: a choice
    vector over all trials stays "choice_pref_align", one over correct trials only becomes
    "choice_pref_align_Response_Correct". Named the same way
    decode_pref_on_choice_axis.get_proj_mode names its projections.

    The run directory is unchanged, since it carries no mode -- the two variants are different file
    names inside it, which is what belief_partitions_io.read_alignment already reads.
    """
    filt_str = belief_partitions_io.get_filter_str(choice_beh_filters)
    return f"{MODE}_{filt_str}" if filt_str else MODE


def prep_behavior_for_feat(raw_beh, feat):
    """
    The feature-dependent, contrast-independent half of what
    decode_belief_partitions.load_session_data ([:32]) does from get_feat_choice_label onward.

    Split out because get_belief_partitions and get_belief_dim_partition both apply row-wise, and
    that is the expensive step in the behavior path -- this way it runs once per (session, feat)
    rather than once per (session, feat, contrast).

    raw_beh is behavioral_utils.load_behavior_from_args output, which is feature-independent and
    is where the shuffle is applied, so it's read once per session.
    """
    beh = behavioral_utils.get_feat_choice_label(raw_beh.copy(), feat)
    beh = behavioral_utils.get_belief_partitions(beh, feat, use_x=True)
    beh = behavioral_utils.get_belief_dim_partition(beh, feat, use_x=True)
    return beh


def prep_behavior(feat_beh, contrast, seed):
    """
    The per-contrast half: assigns `condition`, filters, balances. Gets a copy because
    get_label_by_mode writes `condition` in place for the choice branch.

    One deliberate deviation from load_session_data: the balancing draw is seeded. That call is
    unseeded there, so its subsample differs run to run -- exact trial-level parity with the
    decoding runs was never achievable, and a reproducible run is worth more here.
    """
    beh = behavioral_utils.get_label_by_mode(feat_beh.copy(), contrast["mode"])
    beh = behavioral_utils.filter_behavior(beh, contrast["beh_filters"])
    if BALANCE_CONDITIONS:
        beh = behavioral_utils.balance_trials_by_condition(beh, condition_columns=["condition"], seed=seed)
    return beh


def pivot_frs(frs):
    """
    Reshapes a session's firing rates into a trials x units x time bins array.

    Sorting by [TrialNumber, PseudoUnitID, TimeBins] and reshaping is valid only on a complete
    grid, so that's asserted and a pivot_table fallback used otherwise. Returns the array along
    with the trial numbers and the ascending PseudoUnitIDs its axes correspond to.
    """
    frs = frs.sort_values(["TrialNumber", "PseudoUnitID", "TimeBins"])
    trial_ids = frs.TrialNumber.unique()
    unit_ids = np.sort(frs.PseudoUnitID.unique())
    n_bins = frs.TimeBins.nunique()
    expected = len(trial_ids) * len(unit_ids) * n_bins
    if len(frs) == expected:
        X = frs[DATA_MODE].to_numpy().reshape(len(trial_ids), len(unit_ids), n_bins)
    else:
        print(f"incomplete grid ({len(frs)} rows, expected {expected}), falling back to pivot_table", flush=True)
        pivoted = frs.pivot_table(index="TrialNumber", columns=["PseudoUnitID", "TimeBins"], values=DATA_MODE)
        # pivot_table sorts both index and columns, matching trial_ids / unit_ids order above
        trial_ids = pivoted.index.to_numpy()
        X = pivoted.to_numpy().reshape(len(trial_ids), len(unit_ids), n_bins)
        # genuinely missing (trial, unit, bin) cells would silently poison every vector this
        # session contributes to, so fail here rather than propagating nans
        if np.isnan(X).any():
            raise ValueError(f"{np.isnan(X).sum()} missing (trial, unit, bin) cells after pivot")
    return X, trial_ids, unit_ids


def population_vector(X, trial_pos, beh, contrast_mode):
    """
    Computes the z-scored mean-difference vector for one contrast, as units x time bins.

    Activity is z-scored per (unit, time bin) over the contrast's own pooled trials, matching
    spike_utils.zscore_frs in using ddof=1 and sending zero-variance units to 0. The pooled mean
    cancels in the difference of condition means, so only the division by sd matters.

    Returns (V, n_high, n_low, n_zero_std), or None if either condition is too small.
    """
    high = MODE_TO_DIRECTION_LABELS[contrast_mode]["high"]
    low = MODE_TO_DIRECTION_LABELS[contrast_mode]["low"]

    idx_high = [trial_pos[t] for t in beh[beh.condition == high].TrialNumber if t in trial_pos]
    idx_low = [trial_pos[t] for t in beh[beh.condition == low].TrialNumber if t in trial_pos]
    if len(idx_high) < MIN_TRIALS_PER_COND or len(idx_low) < MIN_TRIALS_PER_COND:
        return None

    pooled = X[idx_high + idx_low]
    sd = pooled.std(axis=0, ddof=1)
    # zero-variance units get coordinate 0 rather than inf/nan, as zscore_frs does
    inv_sd = np.divide(1.0, sd, out=np.zeros_like(sd), where=sd > 0)
    V = (X[idx_high].mean(axis=0) - X[idx_low].mean(axis=0)) * inv_sd
    return V, len(idx_high), len(idx_low), int((sd == 0).sum())


def load_session_frs(sess_name, args):
    """
    A session's (trials x units x bins array, trial number -> row map, ascending PseudoUnitIDs).

    Split out of vectors_for_session because this half depends on (subject, trial_event, fr_type)
    only -- not on the shuffle, which permutes behavior. That's what lets --run_all read each
    session's pickle once per event and hand the array to all 11 cases, and why sessions are the
    outer loop there: only one session's array is alive at a time.

    Returns None if the session has no firing rates.
    """
    frs = spike_utils.get_frs_from_args(args, sess_name)
    if len(frs) == 0:
        print(f"session {sess_name}: no firing rates, skipping", flush=True)
        return None
    X, trial_ids, unit_ids = pivot_frs(frs)
    return X, {t: i for i, t in enumerate(trial_ids)}, unit_ids


def vectors_for_session(sess_name, args, feats_for_sess, session_frs):
    """
    Both contrasts' vectors for every feature this session is valid for, as a long dataframe of
    session, PseudoUnitID, feat, TimeIdx, v_choice, v_pref.

    session_frs is load_session_frs output for this session. Behavior is read once here, since
    with no subpopulation selection the unit set doesn't depend on the feature.

    A (session, feat) pair contributes to both vectors or to neither: if either contrast is too
    small, the pair is dropped. That's what guarantees the two vectors span an identical unit set.
    """
    raw_beh = behavioral_utils.load_behavior_from_args(sess_name, args)
    if len(raw_beh) == 0:
        print(f"session {sess_name}: no behavior, skipping", flush=True)
        return None, []

    X, trial_pos, unit_ids = session_frs

    # shuffles get their own balancing draws, and a re-run of any one job reproduces exactly
    balance_seed = args.train_test_seed + (args.shuffle_idx or 0)

    res = []
    trial_counts = []
    for feat in feats_for_sess:
        vecs = {}
        feat_counts = []
        feat_beh = prep_behavior_for_feat(raw_beh, feat)
        for name, contrast in args.contrasts.items():
            beh = prep_behavior(feat_beh, contrast, balance_seed)
            out = population_vector(X, trial_pos, beh, contrast["mode"])
            counts = {"session": sess_name, "feat": feat, "contrast": name}
            if out is None:
                # record the failure too, so the drop count is auditable
                counts.update({"n_high": np.nan, "n_low": np.nan, "n_zero_std": np.nan, "dropped": True})
                feat_counts.append(counts)
                continue
            V, n_high, n_low, n_zero_std = out
            vecs[name] = V
            counts.update({"n_high": n_high, "n_low": n_low, "n_zero_std": n_zero_std, "dropped": False})
            feat_counts.append(counts)

        if len(vecs) != len(args.contrasts):
            # dropped from both vectors, so the surviving unit sets stay identical
            print(f"session {sess_name} feat {feat}: too few trials in a condition, dropping", flush=True)
            for c in feat_counts:
                c["dropped"] = True
            trial_counts.extend(feat_counts)
            continue
        trial_counts.extend(feat_counts)

        n_bins = vecs["choice"].shape[1]
        df = pd.DataFrame({
            "PseudoUnitID": np.repeat(unit_ids, n_bins),
            "TimeIdx": np.tile(np.arange(n_bins), len(unit_ids)),
            "v_choice": vecs["choice"].ravel(),
            "v_pref": vecs["pref"].ravel(),
        })
        df["session"] = sess_name
        df["feat"] = feat
        res.append(df)
    if len(res) == 0:
        return None, trial_counts
    return pd.concat(res, ignore_index=True), trial_counts


@functools.lru_cache(maxsize=1)
def get_region_of_unit():
    """
    PseudoUnitID -> region map, resolved the same way the decoding runs select regions
    (spike_utils.get_all_region_units on REGION_LEVEL). Cached because it depends on nothing that
    varies across cases, and --run_all labels once per case.
    """
    region_of = {}
    for region in REGIONS_OF_INTEREST:
        for unit_id in spike_utils.get_all_region_units(REGION_LEVEL, region):
            region_of[unit_id] = region
    return region_of


def label_regions(vectors):
    """
    Adds a `region` column to the per-unit vectors.

    Units outside the 6 regions of interest are left unlabelled: they still count toward
    WHOLE_POP, so the regions do not sum to the whole population.
    """
    vectors["region"] = vectors.PseudoUnitID.map(get_region_of_unit())
    n_labelled = vectors[vectors.region.notna()].PseudoUnitID.nunique()
    print(f"labelled {n_labelled} of {vectors.PseudoUnitID.nunique()} units with a region of interest",
          flush=True)
    return vectors


def compute_alignment(vectors, args):
    """
    Collapses the per-unit vectors to one cosine similarity per (region, feat, TimeIdx), for the
    whole population and each region of interest.
    """
    def align_one(group):
        vc = group.v_choice.to_numpy()
        vp = group.v_pref.to_numpy()
        return pd.Series({
            "cos_sim": classifier_utils.cosine_sim(vc, vp),
            "norm_choice": np.linalg.norm(vc),
            "norm_pref": np.linalg.norm(vp),
            "n_units": len(group),
            "n_sessions": group.session.nunique(),
        })

    res = []
    for region in [WHOLE_POP] + list(REGIONS_OF_INTEREST):
        sub = vectors if region == WHOLE_POP else vectors[vectors.region == region]
        # guard both the empty case (cosine_sim would divide by zero) and the too-few-units case
        sub = sub.groupby(["feat", "TimeIdx"]).filter(lambda g: len(g) >= MIN_UNITS_PER_REGION)
        if len(sub) == 0:
            print(f"region {region}: no (feat, bin) reaches {MIN_UNITS_PER_REGION} units, skipping", flush=True)
            continue
        region_res = sub.groupby(["feat", "TimeIdx"]).apply(align_one).reset_index()
        region_res["region"] = region
        res.append(region_res)

    res = pd.concat(res, ignore_index=True)
    res[["n_units", "n_sessions"]] = res[["n_units", "n_sessions"]].astype(int)
    ti = args.trial_interval
    res["Time"] = (res["TimeIdx"] * ti.interval_size + ti.interval_size - ti.pre_interval) / 1000
    shuffle_str = "_shuffle" if args.shuffle_idx is not None else ""
    res["mode"] = f"{args.mode}{shuffle_str}"
    return res


def get_feat_to_sessions(args):
    """
    Maps each feature to the sessions where it was a rule for at least 3 blocks, per subject.

    This restriction is what guarantees enough trials per condition -- a session where X was
    rarely the rule has almost no High X trials. Returns the map plus the union of all sessions,
    which the session_permute shuffle draws its donor session from.
    """
    feat_to_sessions = {}
    all_sessions = []
    for feat in FEATURES:
        feat_args = copy.deepcopy(args)
        feat_args.feat = feat
        sessions = decode_belief_partitions.find_valid_sessions_for_feat_sub(feat_args)
        feat_to_sessions[feat] = list(sessions.session_name)
        all_sessions.append(sessions)
    all_sessions = pd.concat(all_sessions).drop_duplicates(subset="session_name")
    return feat_to_sessions, all_sessions


def args_for_case(args, shuffle_idx):
    """
    A copy of args for one case, where a case is a shuffle index (None being the true run).
    Everything else about a case -- balancing seed, output file name, shuffled behavior -- follows
    from shuffle_idx, so this is the only thing that varies within an event's grid.
    """
    case_args = copy.deepcopy(args)
    case_args.shuffle_idx = shuffle_idx
    return case_args


def align_for_sub(args, cases):
    """
    Runs one subject's sessions for every case in `cases`, returning {case: [per-unit vector dfs]}
    and {case: [trial count dicts]}.

    Sessions are the outer loop and cases the inner one: a session's firing rates are the same for
    every case, so they're read and pivoted once here and shared. Only one session's array is alive
    at a time, which is what keeps the all-cases run's memory the same as a single case's.
    """
    feat_to_sessions, sub_sessions = get_feat_to_sessions(args)
    sess_to_feats = {
        sess: [f for f in FEATURES if sess in feat_to_sessions[f]]
        for sess in sub_sessions.session_name
    }
    case_args = {case: args_for_case(args, case) for case in cases}
    res = {case: [] for case in cases}
    counts = {case: [] for case in cases}
    for sess_name in sub_sessions.session_name:
        feats_for_sess = sess_to_feats[sess_name]
        print(f"session {sess_name}: {len(feats_for_sess)} valid features", flush=True)
        session_frs = load_session_frs(sess_name, args)
        if session_frs is None:
            continue
        for case in cases:
            vectors, trial_counts = vectors_for_session(
                sess_name, case_args[case], feats_for_sess, session_frs
            )
            counts[case].extend(trial_counts)
            if vectors is not None:
                res[case].append(vectors)
    return res, counts


def align(args, cases):
    """
    Computes and saves every case of one trial_event. Each case's outputs are identical to what a
    single-case run writes -- the sharing is only of the firing rates the cases have in common.
    """
    res = {case: [] for case in cases}
    counts = {case: [] for case in cases}

    def collect(sub_res, sub_counts):
        for case in cases:
            res[case].extend(sub_res[case])
            counts[case].extend(sub_counts[case])

    if args.subject == "both":
        # all_sessions must span both subjects, since the shuffle can draw a donor from either
        both_sessions = []
        for sub in ["SA", "BL"]:
            sub_args = copy.deepcopy(args)
            sub_args.subject = sub
            _, sub_sessions = get_feat_to_sessions(sub_args)
            both_sessions.append(sub_sessions)
        both_sessions = pd.concat(both_sessions).drop_duplicates(subset="session_name")

        for sub in ["SA", "BL"]:
            sub_args = copy.deepcopy(args)
            sub_args.subject = sub
            sub_args.all_sessions = both_sessions
            collect(*align_for_sub(sub_args, cases))
    else:
        _, sub_sessions = get_feat_to_sessions(args)
        args.all_sessions = sub_sessions
        collect(*align_for_sub(args, cases))

    for case in cases:
        save_case(res[case], counts[case], args_for_case(args, case))


def save_case(res, counts, args):
    """
    Collapses one case's per-session vectors to the cosine summary and writes both, plus the trial
    counts, under the case's own directory and file name.
    """
    vectors = pd.concat(res, ignore_index=True)
    vectors = label_regions(vectors)
    summary = compute_alignment(vectors, args)

    save_args = copy.deepcopy(args)
    save_args.base_output_path = OUTPUT_PATH
    output_dir = belief_partitions_io.get_dir_name(save_args)
    shuffle_str = "" if args.shuffle_idx is None else f"_shuffle_{args.shuffle_idx}"
    # args.mode carries the choice filters, so a filtered run sits beside the unfiltered one here
    file_name = f"{args.mode}{shuffle_str}"

    summary.to_pickle(os.path.join(output_dir, f"{file_name}.pickle"))
    # per-unit vectors are saved for shuffles too, at ~5MB (StimOnset) / ~8MB (FeedbackOnsetLong)
    # per run. That's what makes any further population split -- a different region level, a unit
    # subset, unit-count-matched subsampling -- a downstream groupby on both the true run AND its
    # null, rather than a re-run
    vectors.to_pickle(os.path.join(output_dir, f"{file_name}_vectors.pickle"))
    pd.DataFrame(counts).to_pickle(os.path.join(output_dir, f"{file_name}_trial_counts.pickle"))

    print(f"\nsaved {len(summary)} (region, feat, TimeIdx) rows to {output_dir}/{file_name}", flush=True)
    print(summary.groupby("region")[["cos_sim", "n_units"]].mean().to_string(), flush=True)
    whole = summary[summary.region == WHOLE_POP]
    print(f"\nmean {WHOLE_POP} cos_sim over all feats/bins: {whole.cos_sim.mean():.4f}", flush=True)


def process_args(args):
    """
    One job covers all 12 features AND all populations, so feat_idx, region_level and regions are
    unused -- passing them would only shrink the unit set that every reported population is drawn
    from. sig_unit_level is likewise unsupported: the point of this analysis is the full population.
    """
    args.mode = get_mode(args.choice_beh_filters)
    args.contrasts = get_contrasts(args.choice_beh_filters)
    if args.region_level is not None or args.regions is not None or args.sig_unit_level is not None:
        raise ValueError(
            "choice_pref_vector_alignment loads the whole population once and reports every region "
            f"from it (see WHOLE_POP + REGIONS_OF_INTEREST), so leave region_level, regions and "
            "sig_unit_level unset. Any further population split is a groupby on the saved "
            "*_vectors.pickle, for the shuffles as well as the true run."
        )
    if args.run_all and args.shuffle_idx is not None:
        raise ValueError(
            "--run_all covers the true run and every shuffle itself, so leave --shuffle_idx unset. "
            "Use --num_shuffles to change how many shuffles it runs."
        )
    print(f"Aligning {list(args.contrasts)} population vectors", flush=True)
    print(f"Contrasts: {args.contrasts}", flush=True)
    print(f"Storing results under mode {args.mode}", flush=True)
    return args


def get_cases(args):
    """
    The (trial_event, [shuffle_idx]) grid to run, as a list of (event, cases) pairs.

    Without --run_all that's the single case the launcher's job would have run; with it, the whole
    grid the launcher spreads over 22 jobs. Cases of one event are grouped so they can share each
    session's firing rates -- see align_for_sub.
    """
    if not args.run_all:
        return [(args.trial_event, [args.shuffle_idx])]
    cases = [None] + list(range(args.num_shuffles))
    return [(event, cases) for event in args.trial_events]


def main(args):
    args = process_args(args)
    for trial_event, cases in get_cases(args):
        event_args = copy.deepcopy(args)
        event_args.trial_event = trial_event
        event_args.trial_interval = get_trial_interval(trial_event)
        shuffle_str = ", ".join("true" if c is None else f"shuffle {c}" for c in cases)
        print(f"\n=== {args.subject}, {trial_event}: {shuffle_str} "
              f"(method {args.shuffle_method}) ===", flush=True)
        align(event_args, cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(BeliefPartitionConfigs(), parser)
    # not part of BeliefPartitionConfigs, which is shared by every decoding script: only this one
    # can walk its own job grid in-process, since no models are fit and a session's firing rates
    # serve every case. --run_all replaces the 22 jobs slurm_launch_choice_pref_alignment.sh
    # submits, writing the same files; the other two exist to shrink or extend that grid
    parser.add_argument('--run_all', default=False, type=lambda x: bool(strtobool(x)))
    parser.add_argument('--trial_events', default=ALL_TRIAL_EVENTS, type=lambda x: x.split(","))
    parser.add_argument('--num_shuffles', default=NUM_SHUFFLES, type=int)
    # trials the choice vector is built from -- the counterpart to decode_pref_on_choice_axis's
    # --axis_beh_filters. Also not part of BeliefPartitionConfigs: --beh_filters there is one set
    # of filters for one run, and this run has two contrasts with different pools. Parsed as json,
    # same as --beh_filters
    parser.add_argument('--choice_beh_filters', default={}, type=lambda x: json.loads(x))
    args = parser.parse_args()
    main(args)
