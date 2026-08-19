"""
Does one neuron carry both the stimulus code and the belief code, or do disjoint sets of neurons
carry them? This is the measurement of claude_notes/stim_belief_single_unit_anova_lite.md part 4,
and the thing that decides H1 vs H3.

Why it is needed. The population analyses of stim_belief_alignment_updated.md are null everywhere:
the A-vs-B projection sits at chance and cos(v_stim, v_pref) does not separate from its shuffle.
But the cosine is an uncentered correlation of two per-unit contrasts across units,

    cos = sum_u d_s[u] d_p[u] / (||d_s|| ||d_p||)   with  d_s = r_B1 - r_A,  d_p = r_C - r_B2

so writing each unit's contribution as |d_s[u]| |d_p[u]| sign(d_s[u] d_p[u]) shows exactly two ways
for it to vanish, which the cosine cannot tell apart:

    (M) for almost every unit one of the magnitudes is ~0 -- the selective sets are disjoint. H1.
    (S) many units have both magnitudes large but the sign is +1 about half the time -- both codes
        live in the same neurons and cancel in the population average. That is ordinary random-sign
        mixed selectivity, H3, and the projection analysis is null under it too, because a linear
        axis fit to A vs B averages the signs away just as the mean-difference vector does.

The cosine used the sign and nothing else. An ANOVA fraction of variance is a function of the
SQUARED contrast, so it uses the magnitude and nothing else -- the two are a complete decomposition
of the same quantity, which is why this reuses the shipped ANOVA rather than inventing a statistic.
Since H1 and H3 both predict a null cosine, the sign branch never has to be measured separately: a
magnitude statistic above its null IS H3, one at or below its null IS H1.

This script reads what 20260819_run_stim_belief_unit_anova.sh produced -- Run S
(x_Choice_comb_time_fracvar, pool A + B1) and Run P (x_BeliefPartition_comb_time_fracvar, pool
B2 + C) -- and crosses them.

Nothing here is computed on raw eta^2. Per-unit null levels are wildly heterogeneous AND correlated
between the two runs (measured on the nearest existing runs: the per-unit 95th percentile threshold
spans 0.005-0.170 for the belief statistic and 0.0006-0.021 for the choice statistic, correlating at
Spearman 0.48 across items). A unit that is noisy is noisy in both runs, so correlating raw eta^2
would recover firing-rate heterogeneity rather than co-selectivity -- the same shared-denominator
artifact that motivates every deattenuation step in cosine_similarity_debiasing.md. Everything below
runs on the exact per-item permutation p-value instead, which is uniform under the null by
construction, per unit, per window, per feature.

An ITEM is a triple (unit, feature, window). That is the right atom because the A/B/C groups are
defined relative to a specific feature X.

Two statistics, per (region, event, window) -- a time course, not a pooled epoch number:

    rho  = N11 N / (N1. N.1)     co-selectivity over what independence predicts GIVEN the observed
                                 marginals, so it is invariant to the two runs having different
                                 power. Reported as log rho, with all four cell counts alongside so
                                 a reader can see whether N11 is 3 or 300. H1: < 1. H3: > 1.
    tau  = spearman(p_s, p_p)    the unthresholded companion. Thresholding at alpha discards every
                                 item below it, which at these selectivity rates is most of the
                                 signal, so tau is the more sensitive of the two and the one to read
                                 first if N11 comes out small. 0 under the null, > 0 under H3,
                                 <= 0 under H1.

Both are tested against a null built by treating each shuffle as the true run (section 4.3). That is
the only null containing the drift x belief-autocorrelation coupling of
trial_subselection_autocorrelation_matching.md, which the B split does not touch, and it is a JOINT
null rather than two marginal ones because Run S's shuffle j and Run P's shuffle j are the same
circular shift of the same session. The null is not assumed to sit at 0 -- whatever offset drift
leaves is inherited by the shuffle statistics, and the comparison is against that.

    python3 stim_belief_unit_overlap.py --run_all True

Memory stays flat: p-values are accumulated one feature at a time and held as exceedance counts
(uint16), so the J x N array for one event is tens of MB rather than GB.
"""

import argparse
import copy
import os
from distutils.util import strtobool

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

import utils.io_utils as io_utils
import utils.spike_utils as spike_utils

from constants.behavioral_constants import *
from constants.decoding_constants import *
from scripts.anova_analysis.anova_configs import AnovaConfigs, add_defaults_to_parser

OUTPUT_PATH = "/data/patrick_res/anova/stim_belief_unit_overlap"

SUBJECTS = ["SA", "BL"]
ALL_TRIAL_EVENTS = ["StimOnset", "FeedbackOnsetLong"]
NUM_SHUFFLES = 100

# per-item false positive rate for the thresholded statistic. With J = 100 the p-value resolution is
# 0.01, so 0.05 is the workable threshold; alpha = 0.01 needs J >= 200
ALPHA = 0.05

WHOLE_POP = "whole_pop"
REGION_LEVEL = "structure_level2_cleaned"

# a (region, window) cell with fewer items than this gives a meaningless contingency table
MIN_ITEMS = 50

STATISTICS = ["log_rho", "tau"]

# the two runs of the note's section 2.1, keyed by the suffix their columns carry here. conditions /
# beh_filters / b_split_half have to match the launcher exactly, since they are what
# io_utils.get_anova_output_dir resolves the input directory from
RUNS = {
    "s": {
        "conditions": ["Choice"],
        "beh_filters": {"Response": "Correct", "BeliefPartition": "High Not X"},
        "b_split_half": 1,
        "stat_col": "x_Choice_comb_time_fracvar",
        "desc": "stimulus (A vs B1)",
    },
    "p": {
        "conditions": ["BeliefPartition"],
        "beh_filters": {"Response": "Correct", "Choice": "Chose", "BeliefConf": "High"},
        "b_split_half": 2,
        "stat_col": "x_BeliefPartition_comb_time_fracvar",
        "desc": "belief (B2 vs C)",
    },
}

ITEM_KEY = ["PseudoUnitID", "WindowStartMilli"]


def run_args(args, run, feat, shuffle_idx=None):
    """
    A copy of args that resolves to one run's anova output dir and file.

    io_utils.get_anova_output_dir / get_anova_file_name are the authority on where the grid landed,
    so the path is never spelled out here -- conditions, beh_filters and b_split_half are set to the
    launcher's values and the path falls out.
    """
    run_args = copy.deepcopy(args)
    for field in ["conditions", "beh_filters", "b_split_half"]:
        setattr(run_args, field, RUNS[run][field])
    run_args.feat = feat
    run_args.shuffle_idx = shuffle_idx
    return run_args


def read_run_feat(args, run, feat):
    """
    One (run, feature)'s eta^2 as a (J+1, N) array with row 0 the true run, plus the item index.

    Every shuffle is reindexed onto the true run's (unit, window) index, so the rows are aligned and
    a shuffle that is missing a unit shows up as a nan rather than a silent shift. Missing FILES are
    an error: a grid with holes in it would quietly change the null's denominator.
    """
    stat_col = RUNS[run]["stat_col"]

    def read(shuffle_idx):
        a = run_args(args, run, feat, shuffle_idx)
        path = os.path.join(io_utils.get_anova_output_dir(a, make_dir=False),
                            f"{io_utils.get_anova_file_name(a)}.pickle")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"missing {path} -- the grid is incomplete, so the permutation null would be built "
                f"on the wrong number of shuffles. Re-submit the launcher (--skip_existing True "
                f"means it only recomputes what is actually absent)."
            )
        return pd.read_pickle(path).set_index(ITEM_KEY)[stat_col]

    true = read(None)
    # a unit x window can appear once only; a duplicated index would make reindex ambiguous
    if true.index.duplicated().any():
        raise ValueError(f"{run}/{feat}: duplicated (unit, window) in the true run")
    eta = np.empty((args.num_shuffles + 1, len(true)), dtype=np.float64)
    eta[0] = true.to_numpy()
    for j in range(args.num_shuffles):
        eta[j + 1] = read(j).reindex(true.index).to_numpy()

    index = true.index.to_frame(index=False)
    index["feat"] = feat
    return eta, index


def permutation_pvals(eta, loo_matched=True):
    """
    Exact per-item permutation p-values, as integer numerators over a common denominator.

    For shuffle j the comparison is leave-one-out -- against the other J-1 shuffles, never against
    itself -- which is algebraically

        p_j = (1 + #{j' != j : eta_j' >= eta_j}) / (1 + (J-1)) = #{j' : eta_j' >= eta_j} / J

    counting j itself in the numerator. Computed from ranks rather than the J x J pairwise
    comparison, since #{j' : eta_j' >= v} = J - rank_min(v) + 1.

    The true run would otherwise sit on a (J+1)-point grid while the shuffles sit on a J-point one.
    That mismatch is second order -- it only changes the tie structure -- but it costs nothing to
    remove, so with loo_matched the true run is compared against J-1 shuffles too and every run's
    p-values land on the identical grid. loo_matched=False restores the plain (1 + #)/(1 + J).

    Returns (numerator for the true run, (J, N) numerators for the shuffles, denominator).
    """
    true, shuf = eta[0], eta[1:]
    n_shuffles = shuf.shape[0]

    rank_min = rankdata(shuf, method="min", axis=0)
    c_shuf = (n_shuffles - rank_min + 1).astype(np.uint16)

    n_comp = n_shuffles - 1 if loo_matched else n_shuffles
    c_true = (1 + (shuf[:n_comp] >= true).sum(axis=0)).astype(np.uint16)
    return c_true, c_shuf, n_comp + 1


def collect_subject_event(args):
    """
    Every item of one (subject, trial_event), with both runs' calibrated p-values.

    Streams one feature at a time -- that is what keeps memory flat, since a feature's (J+1, N_feat)
    float array is released before the next is read, and only the uint16 counts are kept.

    An item is dropped if eta^2 is non-finite in either run, in the true run or in any shuffle.
    That happens when a unit has zero total variance in a window, so its fraction of variance is
    0/0; such an item has no null distribution and cannot be calibrated.
    """
    index, true_eta, counts = [], {}, {}
    per_run = {run: {"c_true": [], "c_shuf": [], "eta": []} for run in RUNS}
    n_dropped = 0

    for feat in FEATURES:
        etas = {run: read_run_feat(args, run, feat) for run in RUNS}
        # both runs cover the same sessions and therefore the same units, but align defensively:
        # a mismatch here would silently pair unit A's stimulus effect with unit B's belief effect
        idx_s, idx_p = etas["s"][1], etas["p"][1]
        if not idx_s.equals(idx_p):
            raise ValueError(f"{feat}: Run S and Run P disagree on the (unit, window) index")

        valid = np.ones(len(idx_s), dtype=bool)
        for run in RUNS:
            valid &= np.isfinite(etas[run][0]).all(axis=0)
        n_dropped += int((~valid).sum())

        for run in RUNS:
            eta = etas[run][0][:, valid]
            c_true, c_shuf, denom = permutation_pvals(eta, args.loo_matched)
            per_run[run]["c_true"].append(c_true)
            per_run[run]["c_shuf"].append(c_shuf)
            per_run[run]["eta"].append(eta[0])
        counts["denom"] = denom
        index.append(idx_s[valid])
        print(f"  {feat}: {valid.sum()} items ({(~valid).sum()} dropped)", flush=True)

    index = pd.concat(index, ignore_index=True)
    for run in RUNS:
        counts[f"c_true_{run}"] = np.concatenate(per_run[run]["c_true"])
        counts[f"c_shuf_{run}"] = np.concatenate(per_run[run]["c_shuf"], axis=1)
        true_eta[f"eta2_{run}"] = np.concatenate(per_run[run]["eta"])

    index["subject"] = args.subject
    index["trial_event"] = args.trial_event
    print(f"  {args.subject}/{args.trial_event}: {len(index)} items, {n_dropped} dropped as non-finite",
          flush=True)
    return index, counts, true_eta


def label_regions(index, subject):
    """
    Region per item, resolved exactly as io_utils.read_anova_good_units does.

    This inner merge is also what drops bad-region and drifting units, which the raw anova pickles
    still contain -- so it is a filter, not just a label. Units kept but outside REGIONS_OF_INTEREST
    still count toward the whole population, so the regions do not sum to it.
    """
    units = spike_utils.get_good_subject_units(subject)[["PseudoUnitID", REGION_LEVEL]]
    labelled = index.merge(units, on="PseudoUnitID", how="left")
    keep = labelled[REGION_LEVEL].notna().to_numpy()
    print(f"  {subject}: kept {keep.sum()} of {len(keep)} items after the good-units merge "
          f"({labelled.loc[keep, 'PseudoUnitID'].nunique()} units)", flush=True)
    return labelled, keep


def stats_for_items(c_s, c_p, thresh):
    """
    Both statistics for one set of items.

    Works on the p-value NUMERATORS rather than the p-values: the threshold p <= alpha is
    c <= alpha * denom, and Spearman is invariant to the common positive denominator, so dividing
    would only lose precision.
    """
    sel_s, sel_p = c_s <= thresh, c_p <= thresh
    n = len(c_s)
    n11, n1_, n_1 = int((sel_s & sel_p).sum()), int(sel_s.sum()), int(sel_p.sum())

    if n1_ == 0 or n_1 == 0:
        log_rho = np.nan
    elif n11 == 0:
        # genuinely zero overlap: maximally H1, and -inf compares correctly against the null
        log_rho = -np.inf
    else:
        log_rho = float(np.log(n11 * n / (n1_ * n_1)))

    # constant input (every item at the same count) makes the correlation undefined, not zero
    if len(np.unique(c_s)) < 2 or len(np.unique(c_p)) < 2:
        tau = np.nan
    else:
        tau = float(spearmanr(c_s, c_p).statistic)

    return {"log_rho": log_rho, "tau": tau,
            "N11": n11, "N1_": n1_, "N_1": n_1, "N": n}


def compute_stats(index, keep, counts, args):
    """
    Both statistics per (region, window), for the true run and for each shuffle taken as true.

    The item subset of a (region, window) does not depend on the shuffle, so the mask is built once
    and reused across the J+1 evaluations -- which is what makes the null cheap.
    """
    thresh = args.alpha * counts["denom"]
    c_true = {run: counts[f"c_true_{run}"] for run in RUNS}
    c_shuf = {run: counts[f"c_shuf_{run}"] for run in RUNS}
    regions = index[REGION_LEVEL].to_numpy()
    windows = index.WindowStartMilli.to_numpy()

    summary, nulls = [], []
    for region in [WHOLE_POP] + list(REGIONS_OF_INTEREST):
        in_region = keep if region == WHOLE_POP else (keep & (regions == region))
        for window in np.unique(windows):
            mask = in_region & (windows == window)
            if mask.sum() < MIN_ITEMS:
                continue
            true_stats = stats_for_items(c_true["s"][mask], c_true["p"][mask], thresh)
            null_stats = [
                stats_for_items(c_shuf["s"][j][mask], c_shuf["p"][j][mask], thresh)
                for j in range(args.num_shuffles)
            ]

            base = {"region": region, "subject": index.subject.iloc[0],
                    "trial_event": args.trial_event, "WindowStartMilli": int(window),
                    "WindowEndMilli": int(window + args.window_size),
                    "n_units": int(pd.unique(index.PseudoUnitID.to_numpy()[mask]).size)}
            for stat in STATISTICS:
                null_vals = np.array([s[stat] for s in null_stats], dtype=float)
                value = true_stats[stat]
                # nan means the statistic was undefined for that shuffle, which cannot serve as a
                # null replicate. -inf does NOT: it is a shuffle with zero overlap, a real and
                # extreme null draw, and dropping those would bias the null upward
                defined = ~np.isnan(null_vals)
                finite = np.isfinite(null_vals)
                summary.append({
                    **base, "statistic": stat, "value": value,
                    "p_h3": exceedance_p(null_vals[defined], value, greater=True),
                    "p_h1": exceedance_p(null_vals[defined], value, greater=False),
                    # summary only, over the finite draws -- the p-values above use every defined one
                    "null_mean": float(np.mean(null_vals[finite])) if finite.any() else np.nan,
                    "n_null": int(defined.sum()),
                    **{k: true_stats[k] for k in ["N11", "N1_", "N_1", "N"]},
                })
                nulls.extend({**base, "statistic": stat, "shuffle_idx": j, "value": v}
                             for j, v in enumerate(null_vals))
    return pd.DataFrame(summary), pd.DataFrame(nulls)


def exceedance_p(null_vals, value, greater=True):
    """One-sided permutation p-value of `value` against `null_vals`, in the named direction."""
    if np.isnan(value) or len(null_vals) == 0:
        return np.nan
    n_beyond = (null_vals >= value).sum() if greater else (null_vals <= value).sum()
    return float((1 + n_beyond) / (1 + len(null_vals)))


def run_event(args):
    """One trial event: both subjects, pooled, plus each subject on its own."""
    print(f"\n=== {args.trial_event} ===", flush=True)
    per_sub = {}
    for subject in SUBJECTS:
        sub_args = copy.deepcopy(args)
        sub_args.subject = subject
        print(f"\nreading {subject}", flush=True)
        index, counts, true_eta = collect_subject_event(sub_args)
        labelled, keep = label_regions(index, subject)
        per_sub[subject] = (labelled, keep, counts, true_eta)

    # pooled is the headline: a region spans both subjects, and shuffle j is drawn within a session
    # either way, so concatenating items keeps the null valid
    pooled_index = pd.concat([per_sub[s][0] for s in SUBJECTS], ignore_index=True)
    pooled_index["subject"] = "both"
    pooled_keep = np.concatenate([per_sub[s][1] for s in SUBJECTS])
    pooled_counts = {"denom": per_sub[SUBJECTS[0]][2]["denom"]}
    for run in RUNS:
        pooled_counts[f"c_true_{run}"] = np.concatenate([per_sub[s][2][f"c_true_{run}"] for s in SUBJECTS])
        pooled_counts[f"c_shuf_{run}"] = np.concatenate([per_sub[s][2][f"c_shuf_{run}"] for s in SUBJECTS], axis=1)

    summaries, nulls = [], []
    for index, keep, counts in ([(pooled_index, pooled_keep, pooled_counts)]
                                + [(per_sub[s][0], per_sub[s][1], per_sub[s][2]) for s in SUBJECTS]):
        summary, null = compute_stats(index, keep, counts, args)
        summaries.append(summary)
        nulls.append(null)
    summary = pd.concat(summaries, ignore_index=True)
    null = pd.concat(nulls, ignore_index=True)

    # the true run's per-item table, so any further breakdown -- one feature, a unit subset, a
    # different region level -- is a downstream groupby rather than a re-read of the 9 GB grid
    items = pd.concat([per_sub[s][0].assign(
        p_s=per_sub[s][2]["c_true_s"] / per_sub[s][2]["denom"],
        p_p=per_sub[s][2]["c_true_p"] / per_sub[s][2]["denom"],
        eta2_s=per_sub[s][3]["eta2_s"], eta2_p=per_sub[s][3]["eta2_p"],
        in_good_units=per_sub[s][1],
    ) for s in SUBJECTS], ignore_index=True)

    return summary, null, items


def save(summary, null, items, args):
    os.makedirs(args.output_path, exist_ok=True)
    event = args.trial_event
    summary.to_pickle(os.path.join(args.output_path, f"{event}_overlap.pickle"))
    null.to_pickle(os.path.join(args.output_path, f"{event}_null.pickle"))
    items.to_pickle(os.path.join(args.output_path, f"{event}_item_pvals.pickle"))
    print(f"\nsaved {len(summary)} (subject, region, window, statistic) rows to "
          f"{args.output_path}/{event}_*", flush=True)

    whole = summary[(summary.region == WHOLE_POP) & (summary.subject == "both")]
    for stat in STATISTICS:
        sub = whole[whole.statistic == stat].sort_values("WindowStartMilli")
        cols = ["WindowStartMilli", "value", "null_mean", "p_h3", "p_h1", "N11", "N1_", "N_1", "N"]
        print(f"\n--- {event}, both subjects, whole population: {stat} ---")
        print(sub[cols].to_string(index=False))

    # the selection rates are the calibration check: the shuffles must sit at alpha by construction,
    # and the true run above it if there is any real selectivity to find
    print(f"\nselection rate at alpha={args.alpha} (true run, whole pop): "
          f"stim {(whole.N1_ / whole.N).mean():.4f}, belief {(whole.N_1 / whole.N).mean():.4f}")


def main(args):
    args.trial_interval = get_trial_interval(args.trial_event)
    events = ALL_TRIAL_EVENTS if args.run_all else [args.trial_event]
    for event in events:
        event_args = copy.deepcopy(args)
        event_args.trial_event = event
        event_args.trial_interval = get_trial_interval(event)
        save(*run_event(event_args), event_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(AnovaConfigs(), parser)
    # --run_all walks both events in one process. There is no per-feature or per-region job here:
    # the accumulation is per feature and every region is a mask over the same arrays
    parser.add_argument("--run_all", default=False, type=lambda x: bool(strtobool(x)))
    parser.add_argument("--num_shuffles", default=NUM_SHUFFLES, type=int)
    parser.add_argument("--alpha", default=ALPHA, type=float)
    parser.add_argument("--loo_matched", default=True, type=lambda x: bool(strtobool(x)))
    parser.add_argument("--output_path", default=OUTPUT_PATH, type=str)
    args = parser.parse_args()
    # the two runs' anova dirs are resolved from these, so they must match the launcher
    args.window_size = 500
    args.use_x = True
    args.shuffle_method = "circular_shift"
    main(args)
