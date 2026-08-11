"""
Significance of the choice/preference population-vector alignment, per timepoint.

For each event, tests whether cos(v_choice, v_pref) sits above its own session-permute shuffle:
    p = P(mean(true*) - mean(shuffle*) >= mean(true) - mean(shuffle))   [one sided permutation]
the same test compute_p_vals_for_decoders.py applies to the decoders, see
stats_utils.compute_p_for_decoding_by_time -- just on "cos_sim" instead of "Accuracy".

Reads what scripts/pseudo_decoding/belief_partitions/choice_pref_vector_alignment.py wrote and
writes the p-values back into each run dir as
    {mode}_pvals.pickle
with columns Time, TimeIdx, p, region -- the whole population plus each region of interest, each
tested against its own shuffle. The Time/TimeIdx/p schema matches what plot_sig_bars consumes, so
callers filter by region and pass the rest straight through.

mode carries the alignment run's --choice_beh_filters, so this script takes that flag too and must
be passed the same value the alignment run was: left unset it tests the all-trials choice vector,
'{"Response": "Correct"}' the correct-only one, whose files sit beside it in the same run dir.

Regions are comparable in significance but NOT in the magnitude of cos_sim: a region has far fewer
units (15-141 per feature vs 382-681 for the whole population), and fewer units means more
attenuation, so a smaller cos_sim in a small region is not evidence of weaker alignment. A regional
dissociation claim needs unit-count-matched subsampling first -- see choice_pref_axis_geometry.md
shared caveat 3, and cosine_similarity_debiasing.md for the attenuation itself.

Note on what the shuffle is for here: unlike the decoding runs, it is not correcting a bias. The
shared-trial cross-term between the two vectors cancels (both preference conditions sit inside the
same choice condition with opposite signs), so the shuffle is expected to sit at ~0. It is the
significance test, and a diagnostic for the one thing the cancellation does not cover -- neural
drift interacting with belief's block autocorrelation. See
claude_notes/cosine_similarity_debiasing.md section 2.

No region loop: the analysis runs on the whole population only.
"""

import os

import utils.stats_utils as stats_utils
from constants.behavioral_constants import *
from constants.decoding_constants import *
from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import *
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
from scripts.pseudo_decoding.belief_partitions.choice_pref_vector_alignment import (
    OUTPUT_PATH, WHOLE_POP, get_mode,
)

import argparse
import json
import pandas as pd
from tqdm import tqdm

NUM_SHUFFLES = 10
TRIAL_EVENTS = ["StimOnset", "FeedbackOnsetLong"]


def run_event(trial_event, choice_beh_filters):
    """
    One p value per (region, timepoint). Each region is tested against its OWN shuffle, so the
    regions are directly comparable in significance -- but NOT in the magnitude of cos_sim, which
    is attenuated more in regions with fewer units. See the module docstring.
    """
    print(f"computing p vals for {trial_event}")

    args = argparse.Namespace(**BeliefPartitionConfigs()._asdict())
    args.subject = "both"
    # picks which alignment run is read, and names this script's output alongside it
    args.mode = get_mode(choice_beh_filters)
    args.trial_event = trial_event
    args.base_output_path = OUTPUT_PATH
    # every population lives in one run dir; region is a column, not a separate run
    args.region_level = None
    args.regions = None
    args.sig_unit_level = None
    args.beh_filters = {}

    res = belief_partitions_io.read_alignment(args, num_shuffles=NUM_SHUFFLES)

    p_vals = []
    for region in res.region.unique():
        region_res = res[res.region == region].copy()
        region_p = stats_utils.compute_p_for_decoding_by_time(region_res, args, val_col="cos_sim")
        region_p["region"] = region
        p_vals.append(region_p)

        is_shuf = region_res["mode"].str.contains("shuffle")
        true_mean, shuffle_mean = region_res[~is_shuf].cos_sim.mean(), region_res[is_shuf].cos_sim.mean()
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
    # must match the alignment run being tested, see the module docstring
    parser.add_argument('--choice_beh_filters', default={}, type=lambda x: json.loads(x))
    args = parser.parse_args()

    events = TRIAL_EVENTS if args.trial_event is None else [args.trial_event]
    for trial_event in tqdm(events):
        run_event(trial_event, args.choice_beh_filters)


if __name__ == "__main__":
    main()
