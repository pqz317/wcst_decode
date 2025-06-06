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
from constants.behavioral_constants import *
import argparse
from constants.decoding_constants import *
from scipy.signal import welch
from tqdm import tqdm

"""
Filter units based off neuronal drift,
Define a threshold based of density of low frequency regimes across the session
If more than 10% of PSD lies in low frequency components (period of > 100 trials)
Then include into list of drift_units
"""

OUTPUT_DIR = "/data/patrick_res/firing_rates/{sub}"

def get_frs(args, session):
    trial_interval = args.trial_interval
    spikes_path = SESS_SPIKES_PATH.format(
        sub=args.subject,
        sess_name=session, 
        fr_type="firing_rates",
        pre_interval=trial_interval.pre_interval, 
        event=trial_interval.event, 
        post_interval=trial_interval.post_interval, 
        interval_size=trial_interval.interval_size
    )
    frs = pd.read_pickle(spikes_path)
    frs["PseudoUnitID"] = int(session) * 100 + frs.UnitID.astype(int)
    return frs

def compute_avg_frs(args, unit_id, all_beh):
    session = unit_id // 100
    all_frs = []
    beh = all_beh[all_beh.session == session]
    for event_idx, event in enumerate(["StimOnset", "FeedbackOnsetLong"]):
        args.trial_event = event
        args.trial_interval = get_trial_interval(event)
        frs = get_frs(args, session)
        frs = frs[frs.TrialNumber.isin(beh.TrialNumber)]
        frs = frs[frs.PseudoUnitID == unit_id]
        all_frs.append(frs)
    all_frs = pd.concat(all_frs)
    avg_frs = all_frs.groupby(["TrialNumber"]).FiringRate.mean().reset_index(name="AvgFiringRate")
    return avg_frs

def get_psd(unit_data, args):
    frequencies, psd = welch(unit_data['AvgFiringRate'], fs=1)
    psd_norm = psd / np.sum(psd)
    mean = np.dot(psd_norm, frequencies) / len(psd_norm)

    low_freq_idxs = np.nonzero(frequencies < args.low_freq_thresh)
    low_freq_dens = np.sum(psd_norm[low_freq_idxs])
    return pd.Series({"psd": psd_norm, "frequencies": frequencies, "mean": mean, "low_freq_dens": low_freq_dens})

def process_session(row, args):
    session = row.session_name
    beh = behavioral_utils.get_valid_belief_beh_for_sub_sess(args.subject, session)
    all_frs = []
    for event_idx, event in enumerate(["StimOnset", "FeedbackOnsetLong"]):
        args.trial_event = event
        args.trial_interval = get_trial_interval(event)
        frs = get_frs(args, session)
        frs = frs[frs.TrialNumber.isin(beh.TrialNumber)]
        all_frs.append(frs)
    all_frs = pd.concat(all_frs)
    avg_frs = all_frs.groupby(["TrialNumber", "PseudoUnitID"]).FiringRate.mean().reset_index(name="AvgFiringRate")
    psds = avg_frs.groupby("PseudoUnitID").apply(lambda x: get_psd(x, args)).reset_index()
    return psds

def filter_units(args):
    sessions = pd.read_pickle(SESSIONS_PATH.format(sub=args.subject))
    unit_psds = pd.concat(sessions.progress_apply(lambda x: process_session(x, args), axis=1).values)
    drifting_units = unit_psds[unit_psds.low_freq_dens > args.density_thresh]
    print(f"{len(drifting_units)} drifting units out of {len(unit_psds)}")
    output_path = os.path.join(OUTPUT_DIR.format(sub=args.subject), "drifting_units.pickle")
    print(f"Saving to {output_path}")
    if not args.dry_run:
        drifting_units[["PseudoUnitID"]].to_pickle(output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', default="SA", type=str)
    parser.add_argument('--dry_run', default=True, type=lambda x: bool(strtobool(x)))
    parser.add_argument('--low_freq_thresh', default=0.01, type=float)
    parser.add_argument('--density_thresh', default=0.1, type=float)
    args = parser.parse_args()

    print(f"Running in dry run {args.dry_run}")
    tqdm.pandas()
    filter_units(args)

if __name__ == "__main__":
    main()