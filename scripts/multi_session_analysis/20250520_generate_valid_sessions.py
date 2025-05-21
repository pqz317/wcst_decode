import glob
import os
from datetime import datetime
import pandas as pd
from spike_tools import (
    general as spike_general,
    analysis as spike_analysis,
)
import argparse
from distutils.util import strtobool

NUM_UNITS = 1
NUM_TRIALS = 1
SA_VALID_SESS_BEFORE = datetime.strptime("20181015", "%Y%m%d").date()
SESS_FOLDER_PATH = "/data/rawdata/sub-{sub}/"
VALID_SESS_PATH = "/data/patrick_res/sessions/{sub}/valid_sessions.pickle"

def grab_sessions_from_fs(subject):
    sess_folder_path = SESS_FOLDER_PATH.format(sub=subject)
    sess_paths = glob.glob(f"{sess_folder_path}/sess-*")
    session_names = [os.path.split(sess_path)[1].split("-")[1] for sess_path in sess_paths]

    rows = []
    for sess_name in session_names:
        if not sess_name.isdigit():
            continue
        # some sessions are recorded on same day, stored as 201808030001
        date = datetime.strptime(sess_name[:8], "%Y%m%d").date()
        rest = sess_name[8:]
        count = int(rest) if rest else 0
        rows.append({
            "session_datetime": date,
            "session_count": count,
            "session_name": sess_name,
        })
    return pd.DataFrame(rows)

def get_num_trials(subject, sess):
    sess_name = sess.session_name
    behavior_path = f"/data/rawdata/sub-{subject}/sess-{sess_name}/behavior/sub-{subject}_sess-{sess_name}_object_features.csv"
    if not os.path.isfile(behavior_path):
        return 0
    beh = pd.read_csv(behavior_path)
    valid_beh = beh[beh.Response.isin(["Correct", "Incorrect"])]   
    return len(valid_beh)

def check_date(sess):
    return sess.session_datetime < SA_VALID_SESS_BEFORE

def get_num_units(subject, sess):
    spike_dir_path = f"/data/rawdata/sub-{subject}/sess-{sess.session_name}/spikes"
    if not os.path.isdir(spike_dir_path):
        return 0
    spike_times = spike_general.get_spike_times(None, subject, sess.session_name, species_dir="/data")
    if spike_times is None: 
        return 0
    return len(spike_times.UnitID.unique())

def filter_sessions(subject, sess):
    sub_sess = sess[(sess.num_trials > NUM_TRIALS) & (sess.num_units > NUM_UNITS)]
    if subject == "SA":
        sub_sess = sub_sess[sub_sess.session_datetime < SA_VALID_SESS_BEFORE]
    return sub_sess

def report_stats(subject, sess):
    print(f"Identified {len(sess)} valid sessions for subject {subject}")
    print(f"Units total: {sess.num_units.sum()}, min: {sess.num_units.min()}, median: {sess.num_units.median()} max: {sess.num_units.max()}")
    print(f"Trials total: {sess.num_trials.sum()}, min: {sess.num_trials.min()}, median: {sess.num_trials.median()} max: {sess.num_trials.max()}")

def main():
    """
    Grabs all sessions from file system, 
    Ensures sessions meet certain criteria, store sessions that fit criteria in a dataframe
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', default='SA', type=str)
    parser.add_argument('--dry_run', default=True, type=lambda x: bool(strtobool(x)))
    args = parser.parse_args()

    subject = args.subject
    sess_df = grab_sessions_from_fs(subject)
    print(f"Found {len(sess_df)} total sessions in fs for subject {subject}")

    sess_df["num_trials"] = sess_df.apply(lambda x: get_num_trials(subject, x), axis=1)
    sess_df["num_units"] = sess_df.apply(lambda x: get_num_units(subject, x), axis=1)

    sess_df = filter_sessions(subject, sess_df)
    report_stats(subject, sess_df)
    print(f"Saving to: {VALID_SESS_PATH.format(sub=subject)}")
    if not args.dry_run:    
        sess_df.to_pickle(VALID_SESS_PATH.format(sub=subject))

if __name__ == "__main__":
    main()