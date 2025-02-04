# to run locally, 
# find correlation to entropy for each 
import argparse
import os
import numpy as np
import pandas as pd
import utils.behavioral_utils as behavioral_utils
import utils.information_utils as information_utils
import utils.spike_utils as spike_utils
import utils.io_utils as io_utils
import utils.glm_utils as glm_utils
from matplotlib import pyplot as plt
import time
from constants.glm_constants import *
from constants.behavioral_constants import *
from constants.decoding_constants import *
from scipy import stats


def calc_and_save_session(row, args):
    start = time.time()
    session = row.session_name

    beh_path = SESS_BEHAVIOR_PATH if args.subject == "SA" else BL_SESS_BEHAVIOR_PATH
    behavior_path = beh_path.format(sess_name=session)
    beh = pd.read_csv(behavior_path)

    # filter trials 
    beh = behavioral_utils.get_valid_trials_blanche(beh)
    feature_selections = behavioral_utils.get_selection_features(beh)
    beh = pd.merge(beh, feature_selections, on="TrialNumber", how="inner")
    beh = behavioral_utils.get_beliefs_per_session(beh, session, subject=args.subject)
    # out, bins = pd.cut(beh["BeliefStateValue"], 10, labels=False, retbins=True)
    # beh["BeliefStateValueBin"] = out
    # beh["BeliefStateValueLabel"] = bins[out]
    beh = behavioral_utils.get_prev_choice_fbs(beh)
    if args.response_cond is not None: 
        if args.trial_event == "StimOnset":
            beh = beh[beh.PrevResponse == args.response_cond]
        elif args.trial_event == "FeedbackOnset":
            beh = beh[beh.Response == args.response_cond]

    trial_interval = get_trial_interval(args.trial_event)

    fr_path = f"/data/patrick_res/firing_rates/SA/{session}_firing_rates_{trial_interval.pre_interval}_{args.trial_event}_{trial_interval.post_interval}_{trial_interval.interval_size}_bins_{NUM_BINS_SMOOTH}_smooth.pickle"

    frs = pd.read_pickle(fr_path)
    agg = frs.groupby(["UnitID", "TrialNumber"]).mean().reset_index()
    reses = []
    for unit in agg.UnitID.unique():
        unit_agg = agg[agg.UnitID == unit]
        merged = pd.merge(unit_agg, beh, on="TrialNumber")
        slope, intercept, r_value, p_value, std_err = stats.linregress(merged["BeliefStateValue"], merged["FiringRate"])
        reses.append({
            "UnitID": unit,
            "session": session,
            "PseudoUnitID": int(session) * 100 + unit,
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value, 
            "p_value": p_value,
            "std_err": std_err
        })
    end = time.time()
    print(f"Session {session} took {(end - start) / 60} minutes")
    return pd.DataFrame(reses)

def process(args):
    sess_path = SA_SESSIONS_PATH if args.subject == "SA" else BL_SESSIONS_PATH
    valid_sess = pd.read_pickle(sess_path)
    all_sess_reses = valid_sess.apply(lambda row: calc_and_save_session(row, args), axis=1)
    all_reses = pd.concat(all_sess_reses.values)

    response_cond_str = "" if args.response_cond is None else f"_{args.response_cond}"
    file_name = f"{args.subject}_belief_state_value_correlations_{args.trial_event}{response_cond_str}"
    all_reses.to_pickle(f"/data/patrick_res/glm_2/{file_name}.pickle")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', default="SA", type=str)
    parser.add_argument('--trial_event', default="StimOnset", type=str)
    parser.add_argument('--response_cond', default=None, type=str)
    args = parser.parse_args()
    
    process(args)

if __name__ == "__main__":
    main()