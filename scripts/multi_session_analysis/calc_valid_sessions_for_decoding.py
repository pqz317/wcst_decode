import glob
import os
from datetime import datetime
import pandas as pd
from spike_tools import (
    general as spike_general,
    analysis as spike_analysis,
)

NUM_NEURONS = 20
NUM_TRIALS = 500
VALID_SESS_BEFORE = datetime.strptime("20181015", "%Y%m%d").date()

def grab_sessions_from_fs():
    # NOTE: this is hacky code to replicate information that should already be stored in a Datajoint table, but don't have access atm
    sess_folder_path = "/data/rawdata/sub-SA/"
    sess_paths = glob.glob(f"{sess_folder_path}/sess-*")
    session_names = [os.path.split(sess_path)[1].split("-")[1] for sess_path in sess_paths]

    rows = []
    for sess_name in session_names:
        if not sess_name.isdigit():
            continue
        # hacky way to grab a datetime
        date = datetime.strptime(sess_name[:8], "%Y%m%d").date()
        rest = sess_name[8:]
        count = int(rest) if rest else 0
        rows.append({
            "session_datetime": date,
            "session_count": count,
            "session_name": sess_name,
        })
    return pd.DataFrame(rows)

def check_num_trials(sess):
    sess_name = sess.session_name
    behavior_path = f"/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
    if not os.path.isfile(behavior_path):
        return False
    beh = pd.read_csv(behavior_path)
    valid_beh = beh[beh.Response.isin(["Correct", "Incorrect"])]   
    return len(valid_beh) > 500

def check_date(sess):
    return sess.session_datetime < VALID_SESS_BEFORE

def check_num_neurons(sess):
    spike_dir_path = f"/data/rawdata/sub-SA/sess-{sess.session_name}/spikes"
    if not os.path.isdir(spike_dir_path):
        return False
    spike_times = spike_general.get_spike_times(None, "SA", sess.session_name, species_dir="/data")
    return len(spike_times.UnitID.unique()) > NUM_NEURONS

def filter_sessions(sess):
    return check_date(sess) and check_num_trials(sess) and check_num_neurons(sess)

def main():
    """
    Grabs all sessions from file system, 
    Ensures sessions meet certain criteria, store sessions that fit criteria in a dataframe
    """
    sess_df = grab_sessions_from_fs()
    print(f"Found {len(sess_df)} total sessions")
    sess_df["valid"] = sess_df.apply(filter_sessions, axis=1)
    valid_sess_df = sess_df[sess_df.valid]
    print(f"Found {len(valid_sess_df)} valid sessions")
    valid_sess_df.to_pickle("/data/patrick_scratch/multi_sess/valid_sessions.pickle")

if __name__ == "__main__":
    main()