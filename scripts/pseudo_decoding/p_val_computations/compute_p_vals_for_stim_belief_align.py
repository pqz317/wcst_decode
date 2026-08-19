"""
Significance of the stimulus/belief population-vector alignment, per timepoint.

For each event, tests whether cos(v_stim, v_pref) sits above its own session-permute shuffle:
    p = P(mean(true*) - mean(shuffle*) >= mean(true) - mean(shuffle))   [one sided permutation]
the same test compute_p_vals_for_decoders.py applies to the decoders, see
stats_utils.compute_p_for_decoding_by_time -- just on a cosine instead of "Accuracy". This is the
A/B/C counterpart of compute_p_vals_for_choice_pref_align.py.

Both of the compute script's estimators are tested: cos_raw, whose denominator is inflated by
finite-trial noise, and cos_cv, whose is cross-validated (cosine_similarity_debiasing.md sections
2a and 4). Same test, same shuffle, one pass each.

Reads what scripts/pseudo_decoding/belief_partitions/stim_belief_vector_alignment.py wrote and
writes the p-values back into each run dir as
    {mode}_pvals.pickle
with columns Time, TimeIdx, p, region, statistic -- the whole population plus each region of
interest, each tested against its own shuffle, for each statistic in STATISTICS. The Time/TimeIdx/p
schema matches what plot_sig_bars consumes, so callers filter by region AND statistic and pass the
rest straight through. Note that filter is new: a caller written against the single-statistic file
will silently read both statistics stacked.

Unlike the choice_pref version there is no --choice_beh_filters: the A/B/C groups fix the pool to
correct trials, so there is exactly one run per event and one mode name. The pool filter is part of
the run directory, which is why POOL_FILTERS is carried on args here.

Regions are comparable in significance but NOT in the magnitude of cos_raw: a region has far fewer
units than the whole population, and fewer units means more attenuation, so a smaller cos_raw in a
small region is not evidence of weaker alignment. Removing exactly that confound is what cos_cv is
for, so its magnitudes are in principle comparable across regions -- but it is also noisier, and a
regional dissociation claim still needs unit-count-matched subsampling first. See
choice_pref_axis_geometry.md shared caveat 3, and cosine_similarity_debiasing.md for the
attenuation itself.

Note on what the shuffle is for here, and how it differs from the choice_pref case. There the
shared-trial cross-term cancels structurally, because both preference conditions sit inside the
same choice condition with opposite signs. The A/B/C design breaks that cancellation -- group B
enters v_stim and v_pref with opposite signs -- so the compute script splits B into disjoint halves
(Issue 1 fix (a) of claude_notes/stim_belief_alignment_updated.md) to zero the cross-term by
construction instead. The shuffle is therefore again expected to sit at ~0, and measuring that it
does is the check that the split worked: without it the null sits near -0.44 (SA/StimOnset,
measured) to -0.65 (the note's balanced-trial arithmetic), which would read H1 as strong
anti-alignment and H2 as roughly zero. A shuffle that is near zero but not at it is the same
neural-drift x belief-autocorrelation term documented in
claude_notes/trial_subselection_autocorrelation_matching.md, which no amount of splitting touches.

Step 6 of that note proposes a per-feature sign-flip enumeration
(stats_utils.compute_p_for_dod_by_time) as the alternative test, with a p-value floor of
1/2^12 = 2.4e-4 at 12 features. This script deliberately runs the same permutation test the shipped
alignment analysis uses, so the two are read the same way.

No region loop: the analysis runs on the whole population only.
"""

import os

import utils.stats_utils as stats_utils
from constants.behavioral_constants import *
from constants.decoding_constants import *
from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import *
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
from scripts.pseudo_decoding.belief_partitions.stim_belief_vector_alignment import (
    OUTPUT_PATH, WHOLE_POP, MODE, POOL_FILTERS,
)

import argparse
import pandas as pd
from tqdm import tqdm

NUM_SHUFFLES = 10
TRIAL_EVENTS = ["StimOnset", "FeedbackOnsetLong"]

# both alignment estimators get the same test, and the output carries a `statistic` column to say
# which is which. cos_raw is kept so the reported numbers stay comparable to the figures already
# produced; cos_cv is the one whose magnitude is interpretable across bins and regions
STATISTICS = ["cos_raw", "cos_cv"]


def run_event(trial_event):
    """
    One p value per (statistic, region, timepoint). Each region is tested against its OWN shuffle,
    so the regions are directly comparable in significance -- but NOT in the magnitude of cos_raw,
    which is attenuated more in regions with fewer units. See the module docstring.
    """
    print(f"computing p vals for {trial_event}")

    args = argparse.Namespace(**BeliefPartitionConfigs()._asdict())
    args.subject = "both"
    args.mode = MODE
    args.trial_event = trial_event
    args.base_output_path = OUTPUT_PATH
    # every population lives in one run dir; region is a column, not a separate run
    args.region_level = None
    args.regions = None
    args.sig_unit_level = None
    # the pool restriction is in the run dir name, so it has to be set to find the run at all
    args.beh_filters = POOL_FILTERS

    res = belief_partitions_io.read_alignment(args, num_shuffles=NUM_SHUFFLES)

    p_vals = []
    for statistic in STATISTICS:
        print(f" {statistic}")
        for region in res.region.unique():
            region_res = res[res.region == region].copy()
            # cos_cv is nan wherever a cross-half squared norm came out <= 0, and the permutation
            # test means over its input, so one nan would make every permuted difference nan and
            # hand back p = 0. Dropping is the only option, but it changes what the test is over,
            # so it is reported rather than done quietly
            n_before = len(region_res)
            region_res = region_res.dropna(subset=[statistic])
            if len(region_res) < n_before:
                print(f"  {region:34s} WARNING dropped {n_before - len(region_res)}/{n_before} "
                      f"(feat, bin, mode) cells with nan {statistic}")
            if len(region_res) == 0:
                print(f"  {region:34s} no usable {statistic}, skipping")
                continue

            region_p = stats_utils.compute_p_for_decoding_by_time(region_res, args, val_col=statistic)
            region_p["region"] = region
            region_p["statistic"] = statistic
            p_vals.append(region_p)

            is_shuf = region_res["mode"].str.contains("shuffle")
            true_mean = region_res[~is_shuf][statistic].mean()
            shuffle_mean = region_res[is_shuf][statistic].mean()
            print(f"  {region:34s} true {true_mean:+.4f}  shuffle {shuffle_mean:+.4f}  "
                  f"gap {true_mean - shuffle_mean:+.4f}  bins p<0.05: {(region_p.p < 0.05).sum()}/{len(region_p)}")
    p_vals = pd.concat(p_vals, ignore_index=True)

    # read_alignment works on a copy, so args.shuffle_idx is still None -> the run dir, not shuffles/
    out_path = os.path.join(belief_partitions_io.get_dir_name(args), f"{args.mode}_pvals.pickle")
    print(f"storing p vals in {out_path}")
    p_vals.to_pickle(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(f'--trial_event', default=None, type=str)
    args = parser.parse_args()

    events = TRIAL_EVENTS if args.trial_event is None else [args.trial_event]
    for trial_event in tqdm(events):
        run_event(trial_event)


if __name__ == "__main__":
    main()
