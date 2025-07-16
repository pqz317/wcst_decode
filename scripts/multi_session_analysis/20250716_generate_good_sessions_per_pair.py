import numpy as np
import pandas as pd
from spike_tools import (
    general as spike_general,
    analysis as spike_analysis,
)
import utils.behavioral_utils as behavioral_utils
import utils.spike_utils as spike_utils
import os
import argparse
from distutils.util import strtobool

NUM_BLOCKS_THRESH = 3
NUM_SESS_THRESH = 10
SESSIONS_PATH = "/data/patrick_res/sessions/{sub}/valid_sessions.pickle"
SESS_BEHAVIOR_PATH = "/data/patrick_res/behavior/{sub}/{sess_name}_object_features.csv"
OUTPUT_PATH = "/data/patrick_res/sessions/{sub}/pairs_at_least_{num_blocks}blocks_{num_sess}sess.pickle"
UNITS_PATH = "/data/patrick_res/firing_rates/{sub}/all_units.pickle"

def load_beh(sub, sess):
    behavior_path = SESS_BEHAVIOR_PATH.format(
        sess_name=sess,
        sub=sub
    )
    beh = pd.read_csv(behavior_path)
    # beh = behavioral_utils.get_valid_trials(beh, sub)
    beh = behavioral_utils.get_valid_trials(beh, "SA")
    beh["session"] = sess
    beh["sub"] = sub
    return beh


def load_all_beh(args):
    if args.subject == "both":
        sa_valid_sess = pd.read_pickle(args.sessions_path.format(sub="SA"))
        sa_behs = pd.concat(sa_valid_sess.apply(lambda x: load_beh("SA", x.session_name), axis=1).values)

        bl_valid_sess = pd.read_pickle(args.sessions_path.format(sub="BL"))
        bl_behs = pd.concat(bl_valid_sess.apply(lambda x: load_beh("BL", x.session_name), axis=1).values)

        return pd.concat((sa_behs, bl_behs))
    return behs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', default="both", type=str)
    parser.add_argument('--sessions_path', default=SESSIONS_PATH, type=str)
    parser.add_argument('--output_path', default=OUTPUT_PATH, type=str)
    parser.add_argument('--units_path', default=UNITS_PATH, type=str)
    parser.add_argument('--num_blocks_thresh', default=NUM_BLOCKS_THRESH, type=int)
    parser.add_argument('--num_sess_thresh', default=NUM_SESS_THRESH)
    parser.add_argument('--dry_run', default=True, type=lambda x: bool(strtobool(x)))

    args = parser.parse_args()

    print(f"Running in dry run {args.dry_run}")
    behs = load_all_beh(args)
    if args.subject == "both":
        sa_units = pd.read_pickle(args.units_path.format(sub="SA"))
        bl_units = pd.read_pickle(args.units_path.format(sub="BL"))
        units = pd.concat((sa_units, bl_units))
    else: 
        units = pd.read_pickle(args.units_path.format(sub=args.subject))
    sess_per_pair = behavioral_utils.get_good_pairs_across_sessions(behs, args.num_blocks_thresh)
    sess_per_pair["n_units"] = sess_per_pair.apply(lambda x: len(units[units.session.isin(x.sessions)]), axis=1)
    sess_per_pair = sess_per_pair[sess_per_pair.num_sessions >= 10]

    to_print = sess_per_pair.sort_values(by="n_units", ascending=False)[["pair", "num_sessions", "n_units", "dim_type"]]
    print(to_print.to_string(index=False))
    print(f"{len(to_print)} pairs")

    if not args.dry_run:
        sess_per_pair.to_pickle(args.output_path.format(
            sub=args.subject,
            num_blocks=args.num_blocks_thresh,
            num_sess=args.num_sess_thresh
        ))


if __name__ == "__main__":
    main()