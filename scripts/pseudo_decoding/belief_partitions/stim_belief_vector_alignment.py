"""
Script for measuring how aligned the stimulus population vector is with the belief population
vector, a time bin at a time, on the whole population.

This is Step 2 of claude_notes/stim_belief_alignment_updated.md, and is the intended replacement
for choice_pref_vector_alignment.py. The difference is structural: that script builds two
independently filtered contrasts, each with its own trial pool and its own z-scoring; this one
takes a single pool of correct trials and splits it three ways, relative to a feature X:

    A: X not chosen, X not preferred     (BeliefPartition == "High Not X", Choice == "Not Chose")
    B: X chosen,     X not preferred     (BeliefPartition == "High Not X", Choice == "Chose")
    C: X chosen,     X preferred         (BeliefPartition == "High X",     Choice == "Chose")

    v_stim = r_B - r_A    # selection of X, holding belief fixed
    v_pref = r_C - r_B    # preference for X, holding selection fixed

and reports cos(v_stim, v_pref). H1 (disjoint stimulus and belief codes) predicts ~0, H2 (aligned
codes, preference scaling the same encoding) predicts > 0.

Group B is split into disjoint halves, and the two vectors are built as

    v_stim = r_B1 - r_A                  v_pref = r_C - r_B2

which is Issue 1 fix (a) of that note, and it is not optional. Both vectors contain r_B with
opposite signs, so sharing it puts E<eps_stim, eps_pref> = -E||e_B||^2 into the numerator: a
structurally negative cross-term worth about cos = -0.65 under the null at these trial counts. H1
would then read as strong anti-alignment and H2 as roughly zero -- exactly backwards. Disjoint
halves make that term exactly zero. Note this problem is new: choice_pref_vector_alignment.py's two
contrasts nest such that the cross-term cancels structurally (claude_notes/
cosine_similarity_debiasing.md section 2b), and the A/B/C design breaks that cancellation.

No trials are subsampled. A mean-difference vector has no need for equal n -- balancing exists to
keep decoder class priors honest -- so all ~294 A, all ~65 B (halved) and all ~55 C trials are
used, which is what keeps each vector's noise power down. The decoding runs of Steps 3-5 balance
as usual; this analysis is deliberately not trial-matched to them.

IMPORTANT interpretation limit: cos_raw is NOT deattenuated. Finite-trial noise inflates both norms
in the denominator, by a factor that differs across time bins, so the number is only meaningful as
a contrast against its shuffle. That is unchanged from the shipped analysis, and halving B makes it
somewhat worse. What this script does instead is save enough to fix it downstream: per-unit group
means r_A, r_B1, r_B2, r_C, their within-group variances, and n per group. The unbiased estimator
of claude_notes/cosine_similarity_debiasing.md section 5,

    sq_stim = ||v_stim||^2 - sum_u ( s2_A[u]/n_A + s2_B1[u]/n_B1 )
    sq_pref = ||v_pref||^2 - sum_u ( s2_B2[u]/n_B2 + s2_C[u]/n_C )
    cos_unb = <v_stim, v_pref> / sqrt( sq_stim * sq_pref )

is therefore a groupby over the saved *_vectors.pickle, on the shuffles as well as the true run,
rather than a re-run. So is any region breakdown or unit subset.

One job per (trial_event, shuffle) is what slurm_launch_stim_belief_alignment.sh submits; passing
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
from distutils.util import strtobool
from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import BeliefPartitionConfigs, add_defaults_to_parser
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
import scripts.pseudo_decoding.belief_partitions.decode_belief_partitions as decode_belief_partitions

# mode results are stored under
MODE = "stim_belief_align"
OUTPUT_PATH = "/data/patrick_res/stim_belief_alignment"

# the single pool all three groups are drawn from. Not configurable: "control for external
# confounds by examining only correct trials" is part of the design, not a variant of it. It is
# still written into the output directory name, so a future variant would sit beside this one
POOL_FILTERS = {"Response": "Correct"}

# the four cells the vectors are built from, after B is halved. Their order is the order the
# per-unit columns appear in
GROUPS = ["A", "B1", "B2", "C"]

# a cell with fewer trials than this can't give a mean and a ddof=1 variance. With the per-feature
# session restriction in force this never fires -- the smallest B half over all 300 (session, feat)
# pairs is 6 -- but the drop is recorded so it would be visible if it did
MIN_TRIALS_PER_GROUP = 2

# populations the cosine is reported for. Every unit's coordinate is computed independently of
# every other unit's, so a region is a subset of the same per-unit vectors rather than a separate
# run -- no region loop over sessions, no refitting, one pass serves all of these
WHOLE_POP = "whole_pop"

# a cosine over a handful of units is meaningless; regions below this for a (feat, bin) are skipped
MIN_UNITS_PER_REGION = 5

REGION_LEVEL = "structure_level2_cleaned"

DATA_MODE = "FiringRate"

# the grid --run_all covers, mirroring slurm_launch_stim_belief_alignment.sh: both events, the
# true run plus 10 shuffles. Overridable with --trial_events / --num_shuffles
ALL_TRIAL_EVENTS = ["StimOnset", "FeedbackOnsetLong"]
NUM_SHUFFLES = 10


def prep_behavior_for_feat(raw_beh, feat):
    """
    The feature-dependent half of what decode_belief_partitions.load_session_data does, plus the
    pool filter, leaving a frame the three group masks apply to directly.

    raw_beh is behavioral_utils.load_behavior_from_args output, which is feature-independent and
    is where the shuffle is applied, so it's read once per session.

    Unlike the choice_pref_align version there is no get_label_by_mode / balance step: the groups
    are explicit masks over one filtered pool, not two contrasts with a `condition` column each.
    """
    beh = behavioral_utils.get_feat_choice_label(raw_beh.copy(), feat)
    beh = behavioral_utils.get_belief_partitions(beh, feat, use_x=True)
    return behavioral_utils.filter_behavior(beh, POOL_FILTERS)


def group_masks(beh):
    """
    The three (Choice x BeliefPartition) cells, as boolean masks over an already-filtered beh.

    Same definitions as claude_notes/stim_belief_group_counts.py, which is where the trial counts
    in the note's Step 1 table come from -- kept in sync by hand, since claude_notes isn't a package.
    """
    return {
        "A": (beh.BeliefPartition == "High Not X") & (beh.Choice == "Not Chose"),
        "B": (beh.BeliefPartition == "High Not X") & (beh.Choice == "Chose"),
        "C": (beh.BeliefPartition == "High X") & (beh.Choice == "Chose"),
    }


def draw_b_split(session, feat, b_trials, seed, shuffle_idx):
    """
    Splits group B's trial numbers into two disjoint halves, B1 and B2 (Issue 1 fix (a)).

    Deterministic in (session, feat, seed, shuffle_idx) and nothing else, so it is reproducible
    without being persisted: Steps 3 and 5 of the note need the choice decoder to train on B1 only
    and the projection to score B2 only, and they get the identical assignment by importing this
    function and calling it with the same arguments.

    Halves are drawn within B rather than as a boolean over every trial in the session, so the two
    are exactly balanced (differing by at most one trial when |B| is odd) rather than binomially
    scattered around |B|/2.
    """
    # a list seed is hashed as SeedSequence entropy, so the four fields can't collide the way an
    # arithmetic combination can. shuffle_idx is offset by 1 to keep the true run's 0 distinct
    rng = np.random.default_rng([
        int(session), FEATURES.index(feat), seed, 0 if shuffle_idx is None else shuffle_idx + 1
    ])
    perm = rng.permutation(np.sort(np.asarray(b_trials)))
    b1, b2 = perm[:len(perm) // 2], perm[len(perm) // 2:]
    assert len(np.intersect1d(b1, b2)) == 0, "B halves overlap"
    assert len(b1) + len(b2) == len(b_trials), "B halves don't partition B"
    return b1, b2


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


def group_stats(X, trial_pos, trials_by_group):
    """
    Per-unit, per-time-bin mean and within-group variance for each of A, B1, B2, C, z-scored.

    Activity is z-scored per (unit, time bin) over the three groups POOLED -- all four cells share
    one sd -- so both vectors live in one metric. That's the whole reason this is a single function
    rather than two calls: the shipped script z-scores each contrast on its own trial pool, which
    the A/B/C design can't do. ddof=1 and zero-variance units sent to 0, matching
    spike_utils.zscore_frs. The pooled mean cancels in both differences and in every variance, so
    only the division by sd matters.

    Returns (means, variances, n, n_zero_std) keyed by group, or None if any cell is too small.
    """
    idx = {}
    for g in GROUPS:
        idx[g] = [trial_pos[t] for t in trials_by_group[g] if t in trial_pos]
        if len(idx[g]) < MIN_TRIALS_PER_GROUP:
            return None

    pooled = X[idx["A"] + idx["B1"] + idx["B2"] + idx["C"]]
    sd = pooled.std(axis=0, ddof=1)
    # zero-variance units get coordinate 0 rather than inf/nan, as zscore_frs does
    inv_sd = np.divide(1.0, sd, out=np.zeros_like(sd), where=sd > 0)

    means, varis, ns = {}, {}, {}
    for g in GROUPS:
        Z = X[idx[g]] * inv_sd
        means[g] = Z.mean(axis=0)
        varis[g] = Z.var(axis=0, ddof=1)
        ns[g] = len(idx[g])
    return means, varis, ns, int((sd == 0).sum())


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
    Per-unit group statistics for every feature this session is valid for, as a long dataframe of
    session, PseudoUnitID, feat, TimeIdx, r_*, s2_*, n_*, v_stim, v_pref.

    session_frs is load_session_frs output for this session. Behavior is read once here, since
    with no subpopulation selection the unit set doesn't depend on the feature.

    v_stim and v_pref are redundant with the means they're differences of; they're stored anyway so
    a downstream groupby reads the same two columns the choice_pref_align notebooks do.
    """
    raw_beh = behavioral_utils.load_behavior_from_args(sess_name, args)
    if len(raw_beh) == 0:
        print(f"session {sess_name}: no behavior, skipping", flush=True)
        return None, []

    X, trial_pos, unit_ids = session_frs

    res = []
    trial_counts = []
    for feat in feats_for_sess:
        beh = prep_behavior_for_feat(raw_beh, feat)
        masks = group_masks(beh)
        # shuffles get their own half-split, and a re-run of any one job reproduces it exactly
        b1, b2 = draw_b_split(
            sess_name, feat, beh[masks["B"]].TrialNumber.to_numpy(),
            args.train_test_seed, args.shuffle_idx,
        )
        trials_by_group = {
            "A": beh[masks["A"]].TrialNumber.to_numpy(),
            "B1": b1,
            "B2": b2,
            "C": beh[masks["C"]].TrialNumber.to_numpy(),
        }

        out = group_stats(X, trial_pos, trials_by_group)
        counts = {"session": sess_name, "feat": feat}
        if out is None:
            # record the failure too, so the drop count is auditable
            print(f"session {sess_name} feat {feat}: too few trials in a group, dropping", flush=True)
            counts.update({f"n_{g}": np.nan for g in GROUPS})
            counts.update({"n_zero_std": np.nan, "dropped": True})
            trial_counts.append(counts)
            continue
        means, varis, ns, n_zero_std = out
        counts.update({f"n_{g}": ns[g] for g in GROUPS})
        counts.update({"n_zero_std": n_zero_std, "dropped": False})
        trial_counts.append(counts)

        n_bins = means["A"].shape[1]
        df = pd.DataFrame({
            "PseudoUnitID": np.repeat(unit_ids, n_bins),
            "TimeIdx": np.tile(np.arange(n_bins), len(unit_ids)),
        })
        for g in GROUPS:
            df[f"r_{g}"] = means[g].ravel()
            df[f"s2_{g}"] = varis[g].ravel()
        df["v_stim"] = (means["B1"] - means["A"]).ravel()
        df["v_pref"] = (means["C"] - means["B2"]).ravel()
        # n varies by session, and the unbiased estimator's variance sums are s2_G[u]/n_G, so
        # carrying n alongside each unit is what makes that a downstream groupby
        for g in GROUPS:
            df[f"n_{g}"] = np.full(len(df), ns[g], dtype=np.int32)
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
        vs = group.v_stim.to_numpy()
        vp = group.v_pref.to_numpy()
        return pd.Series({
            "cos_raw": classifier_utils.cosine_sim(vs, vp),
            "norm_stim": np.linalg.norm(vs),
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

    This restriction is what guarantees enough trials per group -- a session where X was rarely the
    rule has almost no High X trials, so group C empties out. Returns the map plus the union of all
    sessions, which the session_permute shuffle draws its donor session from.
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
    Everything else about a case -- B half-split, output file name, shuffled behavior -- follows
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

    output_dir = belief_partitions_io.get_dir_name(args)
    shuffle_str = "" if args.shuffle_idx is None else f"_shuffle_{args.shuffle_idx}"
    file_name = f"{args.mode}{shuffle_str}"

    summary.to_pickle(os.path.join(output_dir, f"{file_name}.pickle"))
    # per-unit stats are saved for shuffles too. That's what makes any further population split --
    # a different region level, a unit subset, unit-count-matched subsampling -- AND the unbiased
    # estimator of cosine_similarity_debiasing.md section 5 a downstream groupby on both the true
    # run and its null, rather than a re-run
    vectors.to_pickle(os.path.join(output_dir, f"{file_name}_vectors.pickle"))
    pd.DataFrame(counts).to_pickle(os.path.join(output_dir, f"{file_name}_trial_counts.pickle"))

    print(f"\nsaved {len(summary)} (region, feat, TimeIdx) rows to {output_dir}/{file_name}", flush=True)
    print(summary.groupby("region")[["cos_raw", "n_units"]].mean().to_string(), flush=True)
    whole = summary[summary.region == WHOLE_POP]
    print(f"\nmean {WHOLE_POP} cos_raw over all feats/bins: {whole.cos_raw.mean():.4f}", flush=True)


def process_args(args):
    """
    One job covers all 12 features AND all populations, so feat_idx, region_level and regions are
    unused -- passing them would only shrink the unit set that every reported population is drawn
    from. sig_unit_level is likewise unsupported: the point of this analysis is the full population.
    """
    args.mode = MODE
    if args.beh_filters:
        print(f"WARNING: --beh_filters {args.beh_filters} ignored -- the A/B/C groups fix the pool "
              f"to {POOL_FILTERS}", flush=True)
    # carried on args purely so get_dir_name writes the pool restriction into the output path
    args.beh_filters = POOL_FILTERS
    # --base_output_path is left overridable so a test run can write somewhere scratch, but the
    # shared default from BeliefPartitionConfigs means "this analysis's own path"
    if args.base_output_path == BeliefPartitionConfigs().base_output_path:
        args.base_output_path = OUTPUT_PATH
    if args.region_level is not None or args.regions is not None or args.sig_unit_level is not None:
        raise ValueError(
            "stim_belief_vector_alignment loads the whole population once and reports every region "
            f"from it (see WHOLE_POP + REGIONS_OF_INTEREST), so leave region_level, regions and "
            "sig_unit_level unset. Any further population split is a groupby on the saved "
            "*_vectors.pickle, for the shuffles as well as the true run."
        )
    if args.run_all and args.shuffle_idx is not None:
        raise ValueError(
            "--run_all covers the true run and every shuffle itself, so leave --shuffle_idx unset. "
            "Use --num_shuffles to change how many shuffles it runs."
        )
    print("Aligning stimulus (r_B1 - r_A) and belief (r_C - r_B2) population vectors", flush=True)
    print(f"Pool: {POOL_FILTERS}, groups: {GROUPS}", flush=True)
    print(f"Storing results under mode {args.mode} in {args.base_output_path}", flush=True)
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
    # not part of BeliefPartitionConfigs, which is shared by every decoding script: only the
    # alignment scripts can walk their own job grid in-process, since no models are fit and a
    # session's firing rates serve every case. --run_all replaces the 22 jobs
    # slurm_launch_stim_belief_alignment.sh submits, writing the same files; the other two exist to
    # shrink or extend that grid
    parser.add_argument('--run_all', default=False, type=lambda x: bool(strtobool(x)))
    parser.add_argument('--trial_events', default=ALL_TRIAL_EVENTS, type=lambda x: x.split(","))
    parser.add_argument('--num_shuffles', default=NUM_SHUFFLES, type=int)
    args = parser.parse_args()
    main(args)
