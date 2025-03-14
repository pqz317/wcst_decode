"""
Reads in firing rates, generates additional firing rates with trial number (time) regressed out
"""
import numpy as np
import pandas as pd
from spike_tools import (
    general as spike_general,
    analysis as spike_analysis,
)
import utils.behavioral_utils as behavioral_utils
import utils.spike_utils as spike_utils
import os
from distutils.util import strtobool

import argparse
from tqdm import tqdm

"""
Creates firing rates dataframes for all sessions, saves them invdividually
Requires a sessions dataframe, with with column `session_name` that identifies each session

For each session, creates a firing_rates dataframe that should have columns: 
UnitID, TrialNumber, TimeBins, and some data columns, like SpikeCounts or FiringRates
"""

SPECIES = 'nhp'
# SUBJECT = 'SA'

# PRE_INTERVAL = 500
# POST_INTERVAL = 500
# INTERVAL_SIZE = 50
# NUM_BINS_SMOOTH = 1
# EVENT = "FixationOnCross"

# PRE_INTERVAL = 1000
# POST_INTERVAL = 1000
# INTERVAL_SIZE = 100
# NUM_BINS_SMOOTH = 1
# EVENT = "StimOnset"

# PRE_INTERVAL = 1300
# POST_INTERVAL = 1500
# INTERVAL_SIZE = 100
# NUM_BINS_SMOOTH = 1
# EVENT = "FeedbackOnset"

BL_SESSIONS_PATH = "/data/patrick_res/sessions/BL/valid_sessions_61.pickle"
SA_SESSIONS_PATH = "/data/patrick_res/sessions/SA/valid_sessions.pickle"


def transform_fr(row, args):
    sess_name = row.session_name
    print(f"Processing session {sess_name}")
    print("Loading firing rates")
    dir_path = os.path.join(args.base_output_path, args.subject)
    input_file_name = os.path.join(dir_path, f"{sess_name}_firing_rates_{args.pre_interval}_{args.event}_{args.post_interval}_{args.interval_size}_bins_{args.num_bins_smooth}_smooth.pickle")
    try: 
        frs = pd.read_pickle(input_file_name)
    except: 
        raise ValueError(f"file {input_file_name} not found, was the firing rate generated?")
    if args.fr_type == "trial_residual_firing_rates":
        frs = spike_utils.regress_out_trial_number(frs)
    elif args.fr_type == "white_noise_firing_rates":
        frs = spike_utils.white_noise_frs(frs)
    elif args.fr_type == "trial_num_frs":
        frs = spike_utils.trial_num_as_frs(frs)
    elif args.fr_type == "true_pref_belief_frs":
        beh = behavioral_utils.get_valid_belief_beh_for_sub_sess(args.subject, sess_name)
        frs = spike_utils.pref_belief_as_frs(frs, beh)
    else:
        raise ValueError(f"invalid transform {args.fr_type}")

    output_file_name = os.path.join(dir_path, f"{sess_name}_{args.fr_type}_{args.pre_interval}_{args.event}_{args.post_interval}_{args.interval_size}_bins_{args.num_bins_smooth}_smooth.pickle")
    print(f"For sub {args.subject}, session {sess_name}, storing {args.fr_type} of {frs.UnitID.nunique()} units to {output_file_name}", flush=True)
    if not args.dry_run: 
        frs.to_pickle(output_file_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', default="SA", type=str)
    parser.add_argument('--event', default="FeedbackOnset", type=str)
    parser.add_argument('--fr_type', default="trial_residual_firing_rates")
    parser.add_argument('--pre_interval', default=1300, type=int)
    parser.add_argument('--post_interval', default=1500, type=int)
    parser.add_argument('--interval_size', default=100, type=int)
    parser.add_argument('--num_bins_smooth', default=1, type=int)
    parser.add_argument('--base_output_path', default="/data/patrick_res/firing_rates/", type=str)
    parser.add_argument('--dry_run', default=True, type=lambda x: bool(strtobool(x)))
    args = parser.parse_args()

    print(f"Running in dry run: {args.dry_run}")
    if args.subject == "SA":
        valid_sess = pd.read_pickle(SA_SESSIONS_PATH)
    else: 
        valid_sess = pd.read_pickle(BL_SESSIONS_PATH)
    tqdm.pandas()
    valid_sess.progress_apply(lambda row: transform_fr(row, args), axis=1)

if __name__ == "__main__":
    main()