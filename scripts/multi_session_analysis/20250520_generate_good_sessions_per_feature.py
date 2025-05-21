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
SESSIONS_PATH = "/data/patrick_res/sessions/{sub}/valid_sessions.pickle"
SESS_BEHAVIOR_PATH = "/data/patrick_res/behavior/{sub}/{sess_name}_object_features.csv"
OUTPUT_PATH = "/data/patrick_res/sessions/{sub}/feats_at_least_{num_blocks}blocks.pickle"
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
    return beh


def load_all_beh(args):
    valid_sess = pd.read_pickle(args.sessions_path.format(sub=args.subject))
    behs = pd.concat(valid_sess.apply(lambda x: load_beh(args.subject, x.session_name), axis=1).values)
    return behs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', default="SA", type=str)
    parser.add_argument('--sessions_path', default=SESSIONS_PATH, type=str)
    parser.add_argument('--output_path', default=OUTPUT_PATH, type=str)
    parser.add_argument('--units_path', default=UNITS_PATH, type=str)
    parser.add_argument('--num_blocks_thresh', default=NUM_BLOCKS_THRESH, type=int)
    parser.add_argument('--dry_run', default=True, type=lambda x: bool(strtobool(x)))

    args = parser.parse_args()

    print(f"Running in dry run {args.dry_run}")
    behs = load_all_beh(args)
    units = pd.read_pickle(args.units_path.format(sub=args.subject))
    sess_per_feat = behavioral_utils.get_good_sessions_per_rule(behs, args.num_blocks_thresh)
    sess_per_feat["n_units"] = sess_per_feat.apply(lambda x: len(units[units.session.isin(x.sessions)]), axis=1)

    to_print = sess_per_feat.sort_values(by="n_units", ascending=False)[["feat", "num_sessions", "n_units"]]
    print(to_print.to_string(index=False))

    if not args.dry_run:
        sess_per_feat.to_pickle(args.output_path.format(
            sub=args.subject,
            num_blocks=args.num_blocks_thresh
        ))


if __name__ == "__main__":
    main()