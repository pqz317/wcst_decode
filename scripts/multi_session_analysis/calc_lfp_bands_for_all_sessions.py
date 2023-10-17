import numpy as np
import pandas as pd
from spike_tools import (
    general as spike_general,
    analysis as spike_analysis,
)
from lfp_tools import (
    startup_local as startup,
    general,
    analysis
)
import utils.behavioral_utils as behavioral_utils
import utils.spike_utils as spike_utils
import os
import scipy.signal as ss
import utils.behavioral_utils as behavioral_utils

"""
Creates firing rates dataframes for all sessions, saves them invdividually
Requires a sessions dataframe, with with column `session_name` that identifies each session

For each session, creates a firing_rates dataframe that should have columns: 
UnitID, TrialNumber, TimeBins, and some data columns, like SpikeCounts or FiringRates
"""

SPECIES = 'nhp'
SUBJECT = 'SA'
EXP = 'WCST'

PRE_INTERVAL = 0
POST_INTERVAL = 1500
INTERVAL_SIZE = 10
NUM_BINS_SMOOTH = 10
EVENT = "FeedbackOnset"

t_len = 2800
offset = -1300


def gen_lfp_band_data(row, band):
    sess_name = row.session_name
    print(f"Processing {sess_name}, band {band}")
    behavior_path = f"/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
    beh = pd.read_csv(behavior_path)
    try: 
        lfp_files = startup.get_bipole_notch_files(SPECIES, SUBJECT, EXP, sess_name)
    except:
        print(f"No file for sess {sess_name}, skipping")
        return

    valid_beh = behavioral_utils.get_valid_trials(beh)
    df = startup.get_behavior(SPECIES, SUBJECT, EXP, sess_name, import_obj_features=False)
    df_sub = df[
        (df['response'].isin([200,206])) & \
        (df['badGroup']==0) & \
        (df['ignore']==0) & \
        (df['badTrials']==0) & \
        (df['group']>1) & \
        (df['group']<np.max(df.group.values))
    ]
    valid_beh = valid_beh[valid_beh.TrialNumber.isin(df_sub.trial)]
    valid_beh["np_idx"] = np.arange(len(valid_beh))
    t_start = valid_beh.FeedbackOnset.astype(int)
    sig = np.empty((len(lfp_files), t_start.shape[0], t_len))
    for i,f in enumerate(lfp_files):
        print('Load Files', i+1, '/', len(lfp_files))
        lfp = general.open_local_h5py_file(f)
        print('Done loading file, processing')
        if band.split('-')[1]=='nan':
            band_name = "raw"
            lfp_h = analysis.butter_pass_filter(lfp, float(band.split('-')[0]), 1000, 'high')
            lfp_a = np.abs(ss.hilbert(lfp_h))
            sig[i] = analysis.time_Slicer(lfp_a, t_start+offset, t_len)
        else:
            band_name = band
            lfp_h = analysis.butter_pass_filter(lfp, int(band.split('-')[0]), 1000, 'high')
            lfp_b = analysis.butter_pass_filter(lfp_h, int(band.split('-')[1]), 1000, 'low')
            lfp_a = np.abs(ss.hilbert(lfp_b))
            sig[i] = analysis.time_Slicer(lfp_a, t_start+offset, t_len)
    sig_avg = np.mean(np.reshape(sig, (sig.shape[0], sig.shape[1], -1, 100)), axis=3)
    flattened = sig_avg.reshape(-1, 1)
    iterables = [np.arange(sig_avg.shape[0]), np.arange(sig_avg.shape[1]), np.arange(sig_avg.shape[2])]
    index = pd.MultiIndex.from_product(iterables, names=["UnitID", "np_idx", "TimeBinID"])
    df = pd.DataFrame(flattened, index=index)
    df = df.reset_index()
    df = df.rename(columns={0: "Value"})
    df["TimeBins"] = df.TimeBinID / 10
    merged = pd.merge(df, valid_beh, on="np_idx")[["UnitID", "TimeBins", "Value", "TrialNumber"]]
    if len(merged) != len(df):
        raise ValueError(f"dataframe not the same length after merge, was {len(df)}, now is {len(merged)}")

    dir_path = f"/data/patrick_scratch/multi_sess/{sess_name}"
    merged.to_pickle(os.path.join(dir_path, f"{sess_name}_{band}_lfp_{PRE_INTERVAL}_{EVENT}_{POST_INTERVAL}_{INTERVAL_SIZE}_bins.pickle"))



def main():
    valid_sess = pd.read_pickle("/data/patrick_scratch/multi_sess/valid_sessions.pickle")
    bands = startup.get_bands(SPECIES, SUBJECT, EXP)
    bands = [bands[-1]]
    for band in bands:
        for 
        valid_sess.apply(lambda row: gen_lfp_band_data(row, band), axis=1)

if __name__ == "__main__":
    main()