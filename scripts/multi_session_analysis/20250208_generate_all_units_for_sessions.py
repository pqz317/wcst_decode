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

SESSIONS_PATH = "/data/patrick_res/sessions/{sub}/valid_sessions.pickle"
OUTPUT_PATH = "/data/patrick_res/firing_rates/{sub}/all_units.pickle"

def get_units_for_session(row, args):
    sess_name = row.session_name
    sess_units = spike_general.list_session_units(None, args.subject, sess_name, species_dir="/data")
    sess_units["session"] = sess_name
    sess_units["PseudoUnitID"] = int(row.session_name) * 100 + sess_units.UnitID
    if len(sess_units) > 0:
        print(f"session {sess_name}: {sess_units.UnitID.nunique()} units")

    return sess_units


def generate_all_units(args):
    sessions = pd.read_pickle(args.sessions_path.format(sub=args.subject))
    if args.subject == "SA": 
        all_units = spike_utils.get_unit_positions(sessions, fr_path=None)
    else: 
        all_units = pd.concat(sessions.apply(lambda x: get_units_for_session(x, args), axis=1).values)
    print(f"{len(all_units)} units total")
    print(f"across {all_units.session.nunique()} sessions")
    all_units.to_pickle(args.output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', default="SA", type=str)
    parser.add_argument('--sessions_path', default=SESSIONS_PATH, type=str)
    parser.add_argument('--output_path', default=OUTPUT_PATH, type=str )
    args = parser.parse_args()
    generate_all_units(args)

if __name__ == "__main__":
    main()