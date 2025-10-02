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


SESSIONS_PATH = "/data/patrick_res/sessions/{sub}/valid_sessions.pickle"
OUTPUT_PATH = "/data/patrick_res/firing_rates/{sub}/all_units_corrected.pickle"

def get_units_for_session(row, args):
    sess_name = row.session_name
    sess_units = spike_general.list_session_units(None, args.subject, sess_name, species_dir="/data")
    sess_units["session"] = sess_name
    sess_units["PseudoUnitID"] = int(row.session_name) * 100 + sess_units.UnitID
    print(f"session {sess_name}: {sess_units.UnitID.nunique()} units")

    return sess_units


def generate_all_units(args):
    sessions = pd.read_pickle(args.sessions_path.format(sub=args.subject))
    get_manual_regions = args.subject == "SA"  # only get manual regions for SA
    # all_units = spike_utils.get_unit_positions(sessions, args.subject, get_manual_regions, fr_path=None)
    all_units = spike_utils.get_unit_positions(
        sessions, args.subject, get_manual_regions, fr_path=None, 
        sess_info_path="/data/rawdata/sub-{subject}/sess-{session}/session_info_corrected/sub-{subject}_sess-{session}_sessioninfomodified.json"
    )

    print(f"{len(all_units)} units total")
    print(f"across {all_units.session.nunique()} sessions")
    print("unit region stats:")
    print(all_units.groupby("structure_level2_cleaned").PseudoUnitID.nunique().to_csv())

    output_path = args.output_path.format(sub=args.subject)
    print(f"saving to {output_path}")
    if not args.dry_run: 
        all_units.to_pickle(output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', default="SA", type=str)
    parser.add_argument('--sessions_path', default=SESSIONS_PATH, type=str)
    parser.add_argument('--output_path', default=OUTPUT_PATH, type=str)
    parser.add_argument('--dry_run', default=True, type=lambda x: bool(strtobool(x)))

    args = parser.parse_args()

    print(f"Running in dry run {args.dry_run}")
    generate_all_units(args)

if __name__ == "__main__":
    main()