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

That fixes the numerator. The denominator has a separate problem: cos_raw is NOT deattenuated.
Finite-trial noise inflates both measured norms, by a factor that varies with n and therefore
across time bins, features and regions, so cos_raw's SHAPE is distorted and not just its height,
and the number is only readable as a contrast against its own shuffle. Halving B makes it somewhat
worse. Two corrections exist, and this script reports one and saves what the other needs.

cos_cv (claude_notes/cosine_similarity_debiasing.md section 4) is computed here. It halves A and C
as well as B, once per repeat, and builds one pair of vectors per half:

    v_stim^h = r_Bh - r_Ah               v_pref^h = r_Ch - r_Bh

Every product pairs half 1 against half 2, whose trials are disjoint, so the squared norms come out
of cross-half inner products with no noise power in them:

    num     = 1/2 ( <v_stim^1, v_pref^2> + <v_stim^2, v_pref^1> )
    sq_stim = <v_stim^1, v_stim^2>       sq_pref = <v_pref^1, v_pref^2>
    cos_cv  = num / sqrt( sq_stim * sq_pref )

averaged over --num_cv_repeats half-splits and divided once at the end. It needs no noise model and
no independent-trials assumption; it costs half the data behind every group, so sq_* can come out
negative (cos_cv is then nan) and |cos_cv| can exceed 1. Neither is clipped. att_stim_cv /
att_pref_cv = sqrt(sq / ss) are reported alongside, ss being the same quantity measured within a
half instead of across, i.e. the fraction of measured vector length that is signal.

cos_unb (section 5) is the alternative: keep cos_raw's vectors and subtract the noise power
analytically. Not computed here, but the per-unit group means, within-group variances and n per
group it needs are all saved, so

    sq_stim = ||v_stim||^2 - sum_u ( s2_A[u]/n_A + s2_B1[u]/n_B1 )
    sq_pref = ||v_pref||^2 - sum_u ( s2_B2[u]/n_B2 + s2_C[u]/n_C )
    cos_unb = <v_stim, v_pref> / sqrt( sq_stim * sq_pref )

is a groupby over the saved *_vectors.pickle, on the shuffles as well as the true run, rather than
a re-run. So is any region breakdown or unit subset, of either estimator: the per-unit cv_* terms
are saved for the same reason, and summing them over a population and dividing once reproduces
that population's cos_cv.

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

# the three cells before any halving. cos_raw halves only B; cos_cv halves all three, so this is
# what the half-split and the min-trial guard are expressed over
BASE_GROUPS = ["A", "B", "C"]

# a seed field, so the three groups' half-assignments are independent draws rather than the same
# permutation applied three times
GROUP_CODE = {"A": 1, "B": 2, "C": 3}

# the per-unit terms cos_cv is assembled from, summed over whatever population is being reported.
# cv_num / cv_sq_* are cross-half products and carry no noise power; cv_ss_* are self-half products
# and carry signal + noise, which is what makes their ratio an attenuation factor
CV_PIECES = ["cv_num", "cv_sq_stim", "cv_sq_pref", "cv_ss_stim", "cv_ss_pref"]

# half-splits to average cos_cv's three pieces over. The assignment is arbitrary, so repeating it
# and averaging costs only compute and cuts the split-induced variance. Reading each session's
# firing rates dominates the runtime, so 20 is nowhere near the constraint
NUM_CV_REPEATS = 20

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


def draw_half_split(session, feat, trials, seed, shuffle_idx, group="B", repeat=0):
    """
    Splits one base group's trial numbers into two disjoint halves.

    Deterministic in (session, feat, seed, shuffle_idx, group, repeat) and nothing else, so it is
    reproducible without being persisted: Steps 3 and 5 of the note need the choice decoder to
    train on B1 only and the projection to score B2 only, and they get the identical assignment by
    importing draw_b_split below and calling it with the same arguments.

    Halves are drawn within the group rather than as a boolean over every trial in the session, so
    the two are exactly balanced (differing by at most one trial when |G| is odd) rather than
    binomially scattered around |G|/2.

    cos_raw needs one call, on B, and that is what draw_b_split names. cos_cv needs all three of
    A, B and C halved, once per repeat, which is what `group` and `repeat` index. The two agree on
    B at repeat 0 by construction, so cos_cv's first repeat reuses cos_raw's B1/B2 exactly.
    """
    # a list seed is hashed as SeedSequence entropy, so the six fields can't collide the way an
    # arithmetic combination can. shuffle_idx is offset by 1 to keep the true run's 0 distinct
    rng = np.random.default_rng([
        int(session), FEATURES.index(feat), seed,
        0 if shuffle_idx is None else shuffle_idx + 1,
        GROUP_CODE[group], repeat,
    ])
    perm = rng.permutation(np.sort(np.asarray(trials)))
    h1, h2 = perm[:len(perm) // 2], perm[len(perm) // 2:]
    assert len(np.intersect1d(h1, h2)) == 0, f"{group} halves overlap"
    assert len(h1) + len(h2) == len(trials), f"{group} halves don't partition {group}"
    return h1, h2


def draw_b_split(session, feat, b_trials, seed, shuffle_idx):
    """
    Group B's canonical halves, B1 and B2 (Issue 1 fix (a)) -- the assignment v_stim = r_B1 - r_A
    and v_pref = r_C - r_B2 are built from, and the one Steps 3 and 5 reproduce by importing this.
    """
    return draw_half_split(session, feat, b_trials, seed, shuffle_idx, group="B", repeat=0)


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


def pooled_z(X, trial_pos, trials_by_base_group):
    """
    The pooled three-group activity, z-scored per (unit, time bin), plus a trial number -> row map.

    Activity is z-scored over A, B and C POOLED -- every cell and every half shares one sd -- so
    all the vectors built downstream live in one metric. That's the whole reason this is computed
    up front rather than inside each consumer: the shipped script z-scores each contrast on its own
    trial pool, which the A/B/C design can't do, and cos_cv additionally can't afford a half-
    specific sd. ddof=1 and zero-variance units sent to 0, matching spike_utils.zscore_frs. The
    pooled mean cancels in both differences and in every variance, so only the division by sd
    matters.

    Returns (Z, rows, n_zero_std), or None if any base group is too small to halve -- which is the
    "drop the (session, feat) from every vector" rule of cosine_similarity_debiasing.md section 4,
    so cos_raw and cos_cv always span an identical unit set. Over the 300 (session, feat) pairs the
    smallest group is 12 trials, so this never fires.
    """
    pool_trials, pool_pos = [], []
    for g in BASE_GROUPS:
        matched = [t for t in trials_by_base_group[g] if t in trial_pos]
        # 2x, not 1x: a half of this group has to clear MIN_TRIALS_PER_GROUP too
        if len(matched) < 2 * MIN_TRIALS_PER_GROUP:
            return None
        pool_trials.extend(matched)
        pool_pos.extend(trial_pos[t] for t in matched)

    pooled = X[pool_pos]
    sd = pooled.std(axis=0, ddof=1)
    # zero-variance units get coordinate 0 rather than inf/nan, as zscore_frs does
    inv_sd = np.divide(1.0, sd, out=np.zeros_like(sd), where=sd > 0)
    Z = pooled * inv_sd
    return Z, {t: i for i, t in enumerate(pool_trials)}, int((sd == 0).sum())


def rows_for(rows, trials):
    """
    Rows of a pooled_z array for a set of trial numbers, skipping any without firing rates.

    The splits are drawn over behavioral trial numbers rather than over rows, which is what keeps
    them reproducible from behavior alone (see draw_half_split), so the two have to be reconciled
    somewhere. This is that somewhere.
    """
    return [rows[t] for t in trials if t in rows]


def group_stats(Z, rows, trials_by_group):
    """
    Per-unit, per-time-bin mean and within-group variance for each of A, B1, B2, C.

    Feeds cos_raw's v_stim / v_pref, and -- through the saved s2_* and n_* columns -- the
    analytic deattenuation of cosine_similarity_debiasing.md section 5.

    Returns (means, variances, n) keyed by group, or None if any cell is too small.
    """
    means, varis, ns = {}, {}, {}
    for g in GROUPS:
        idx = rows_for(rows, trials_by_group[g])
        if len(idx) < MIN_TRIALS_PER_GROUP:
            return None
        means[g] = Z[idx].mean(axis=0)
        varis[g] = Z[idx].var(axis=0, ddof=1)
        ns[g] = len(idx)
    return means, varis, ns


def cv_pieces(Z, rows, trials_by_base_group, session, feat, args, n_repeats):
    """
    Per-unit terms of the cross-validated cosine, averaged over n_repeats half-splits.

    Section 4 of cosine_similarity_debiasing.md. Each repeat halves ALL THREE groups, giving
    A1/A2, B1/B2, C1/C2, and builds one pair of vectors per half:

        v_stim^h = r_Bh - r_Ah               v_pref^h = r_Ch - r_Bh

    The two vectors of a half share the same B_h, so they are never multiplied together; every
    product below pairs half 1 against half 2, whose trials are disjoint and whose sampling errors
    are therefore independent. That is what makes E<v_stim^1, v_pref^2> = <v_stim, v_pref> with no
    cross-term AND E<v_stim^1, v_stim^2> = ||v_stim||^2 with no noise inflation -- the second is
    what cos_raw's denominator gets wrong and what this estimator exists to fix, without needing a
    noise model or the independent-trials assumption section 5's cos_unb rests on.

    The price is half the data behind every group, so cv_sq_* can come out negative and |cos_cv|
    can exceed 1. Both are expected of an unbiased ratio estimator and are NOT clipped here.

    cv_ss_* are the self-products, carrying signal + noise where cv_sq_* carries signal alone, so
    sqrt(cv_sq / cv_ss) is a cross-validated attenuation factor -- the fraction of measured vector
    length that is real signal, equivalently the vector's split-half reliability. Measured at half
    data, where noise power is ~2x the full-data vectors', so the full-data factor is
    sqrt(sq / (sq + (ss - sq) / 2)).

    Everything is per unit and per time bin, and the average over repeats commutes with the sum
    over units, so summing these columns over any population and dividing once at the end gives
    that population's cos_cv -- which keeps a region breakdown a downstream groupby, exactly as it
    is for cos_raw and cos_unb.

    Returns (pieces keyed by CV_PIECES, repeats used), or (None, 0) if no repeat was usable.
    """
    acc = {k: 0.0 for k in CV_PIECES}
    used = 0
    for repeat in range(n_repeats):
        halves = {}
        for g in BASE_GROUPS:
            h1, h2 = draw_half_split(
                session, feat, trials_by_base_group[g], args.train_test_seed, args.shuffle_idx,
                group=g, repeat=repeat,
            )
            idx1, idx2 = rows_for(rows, h1), rows_for(rows, h2)
            if min(len(idx1), len(idx2)) < MIN_TRIALS_PER_GROUP:
                halves = None
                break
            halves[g] = (Z[idx1].mean(axis=0), Z[idx2].mean(axis=0))
        # pooled_z's 2x guard makes this unreachable on this data, but a repeat that can't be built
        # is dropped rather than poisoning the average
        if halves is None:
            continue

        vs = [halves["B"][h] - halves["A"][h] for h in (0, 1)]
        vp = [halves["C"][h] - halves["B"][h] for h in (0, 1)]
        # both orderings of (half, vector), which uses every trial in both roles and is why
        # crossnobis averages over all ordered pairs of distinct partitions
        acc["cv_num"] += 0.5 * (vs[0] * vp[1] + vs[1] * vp[0])
        acc["cv_sq_stim"] += vs[0] * vs[1]
        acc["cv_sq_pref"] += vp[0] * vp[1]
        acc["cv_ss_stim"] += 0.5 * (vs[0] ** 2 + vs[1] ** 2)
        acc["cv_ss_pref"] += 0.5 * (vp[0] ** 2 + vp[1] ** 2)
        used += 1

    if used == 0:
        return None, 0
    return {k: v / used for k, v in acc.items()}, used


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
    session, PseudoUnitID, feat, TimeIdx, r_*, s2_*, n_*, v_stim, v_pref, cv_*.

    session_frs is load_session_frs output for this session. Behavior is read once here, since
    with no subpopulation selection the unit set doesn't depend on the feature.

    v_stim and v_pref are redundant with the means they're differences of; they're stored anyway so
    a downstream groupby reads the same two columns the choice_pref_align notebooks do. The cv_*
    columns are not redundant with anything: their half-means are drawn per repeat and averaged
    away, so they are the only record of the cross-validated estimator.
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
        trials_by_base_group = {g: beh[masks[g]].TrialNumber.to_numpy() for g in BASE_GROUPS}
        # shuffles get their own half-split, and a re-run of any one job reproduces it exactly
        b1, b2 = draw_b_split(
            sess_name, feat, trials_by_base_group["B"], args.train_test_seed, args.shuffle_idx,
        )
        trials_by_group = {
            "A": trials_by_base_group["A"],
            "B1": b1,
            "B2": b2,
            "C": trials_by_base_group["C"],
        }

        counts = {"session": sess_name, "feat": feat}
        pooled = pooled_z(X, trial_pos, trials_by_base_group)
        out = None if pooled is None else group_stats(pooled[0], pooled[1], trials_by_group)
        if out is None:
            # record the failure too, so the drop count is auditable
            print(f"session {sess_name} feat {feat}: too few trials in a group, dropping", flush=True)
            counts.update({f"n_{g}": np.nan for g in GROUPS})
            counts.update({"n_zero_std": np.nan, "n_cv_repeats": 0, "dropped": True})
            trial_counts.append(counts)
            continue
        Z, rows, n_zero_std = pooled
        means, varis, ns = out
        pieces, n_cv = cv_pieces(
            Z, rows, trials_by_base_group, sess_name, feat, args, args.num_cv_repeats
        )
        counts.update({f"n_{g}": ns[g] for g in GROUPS})
        counts.update({"n_zero_std": n_zero_std, "n_cv_repeats": n_cv, "dropped": False})
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
        # NaN rather than absent when the cv path is off (--num_cv_repeats 0), so the saved schema
        # doesn't depend on the flag
        for k in CV_PIECES:
            df[k] = np.nan if pieces is None else pieces[k].ravel()
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

    Two estimators, sharing a numerator design (B is halved either way) and differing in the
    denominator: cos_raw divides by the measured norms, which finite-trial noise inflates by an
    amount that varies with n and therefore across bins, features and regions; cos_cv divides by
    cross-half inner products, which estimate the noiseless squared norms directly. See
    cosine_similarity_debiasing.md sections 2a and 4.
    """
    def align_one(group):
        vs = group.v_stim.to_numpy()
        vp = group.v_pref.to_numpy()
        # summing the per-unit terms and dividing ONCE is the point: a mean of per-unit or
        # per-feature ratios would be dominated by near-zero denominators
        num, sq_stim, sq_pref, ss_stim, ss_pref = (group[k].sum() for k in CV_PIECES)
        # both factors have to be positive, not just their product -- two negative squared norms
        # multiply to a positive one and would hand back a sign-flipped cosine
        cv_ok = sq_stim > 0 and sq_pref > 0
        return pd.Series({
            "cos_raw": classifier_utils.cosine_sim(vs, vp),
            "cos_cv": num / np.sqrt(sq_stim * sq_pref) if cv_ok else np.nan,
            "norm_stim": np.linalg.norm(vs),
            "norm_pref": np.linalg.norm(vp),
            "cv_num": num,
            "cv_sq_stim": sq_stim,
            "cv_sq_pref": sq_pref,
            "cv_ss_stim": ss_stim,
            "cv_ss_pref": ss_pref,
            # fraction of measured vector length that is signal, at half data. Answers what the
            # cosine alone cannot: whether cos ~ 0 means orthogonal codes or no measurable signal
            "att_stim_cv": np.sqrt(max(sq_stim, 0) / ss_stim) if ss_stim > 0 else np.nan,
            "att_pref_cv": np.sqrt(max(sq_pref, 0) / ss_pref) if ss_pref > 0 else np.nan,
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
    # a different region level, a unit subset, unit-count-matched subsampling -- for cos_raw, for
    # cos_cv (sum the cv_* columns, divide once) AND for the unbiased estimator of
    # cosine_similarity_debiasing.md section 5 a downstream groupby on both the true run and its
    # null, rather than a re-run
    vectors.to_pickle(os.path.join(output_dir, f"{file_name}_vectors.pickle"))
    pd.DataFrame(counts).to_pickle(os.path.join(output_dir, f"{file_name}_trial_counts.pickle"))

    print(f"\nsaved {len(summary)} (region, feat, TimeIdx) rows to {output_dir}/{file_name}", flush=True)
    cols = ["cos_raw", "cos_cv", "att_stim_cv", "att_pref_cv", "n_units"]
    print(summary.groupby("region")[cols].mean().to_string(), flush=True)
    whole = summary[summary.region == WHOLE_POP]
    print(f"\nmean {WHOLE_POP} cos_raw over all feats/bins: {whole.cos_raw.mean():.4f}", flush=True)
    print(f"mean {WHOLE_POP} cos_cv  over all feats/bins: {whole.cos_cv.mean():.4f}", flush=True)
    # the headline diagnostic for whether the half-data estimator is usable at all: cos_cv is only
    # defined where both cross-half squared norms come out positive
    bad_stim, bad_pref = whole.cv_sq_stim <= 0, whole.cv_sq_pref <= 0
    print(f"{WHOLE_POP} (feat, bin) cells with cos_cv undefined: {(bad_stim | bad_pref).sum()}/{len(whole)} "
          f"(sq_stim<=0 only {(bad_stim & ~bad_pref).sum()}, sq_pref<=0 only {(~bad_stim & bad_pref).sum()}, "
          f"both {(bad_stim & bad_pref).sum()})", flush=True)


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
    if args.num_cv_repeats > 0:
        print(f"Cross-validated cosine over {args.num_cv_repeats} half-splits of A, B and C", flush=True)
    else:
        print("Cross-validated cosine DISABLED (--num_cv_repeats 0): cos_cv will be nan", flush=True)
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
    # cos_cv's half-splits to average over. Deliberately not num_splits from BeliefPartitionConfigs:
    # that one only reaches the output directory name when splitter == "kfold", so reusing it here
    # would silently write two different runs to the same path
    parser.add_argument('--num_cv_repeats', default=NUM_CV_REPEATS, type=int)
    args = parser.parse_args()
    main(args)
