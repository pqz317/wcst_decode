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
from scipy import stats


def calc_and_save_session(row):
    start = time.time()
    session = row.session_name
    # behavior_path = SESS_BEHAVIOR_PATH.format(sess_name=session)
    behavior_path = BL_SESS_BEHAVIOR_PATH.format(sess_name=session)
    beh = pd.read_csv(behavior_path)

    # filter trials 
    beh = behavioral_utils.get_valid_trials_blanche(beh)
    feature_selections = behavioral_utils.get_selection_features(beh)
    beh = pd.merge(beh, feature_selections, on="TrialNumber", how="inner")
    beh = behavioral_utils.get_beliefs_per_session(beh, session, subject="BL")
    # out, bins = pd.cut(beh["BeliefStateValue"], 10, labels=False, retbins=True)
    # beh["BeliefStateValueBin"] = out
    # beh["BeliefStateValueLabel"] = bins[out]

    beh = behavioral_utils.get_prev_choice_fbs(beh)
    # beh = beh[beh.PrevResponse == "Incorrect"]

    # fr_path = f"/data/patrick_res/firing_rates/{session}_firing_rates_{PRE_INTERVAL}_{EVENT}_{POST_INTERVAL}_{INTERVAL_SIZE}_bins_{NUM_BINS_SMOOTH}_smooth.pickle"
    fr_path = f"/data/patrick_res/firing_rates/BL_{session}_firing_rates_{PRE_INTERVAL}_{EVENT}_{POST_INTERVAL}_{INTERVAL_SIZE}_bins_{NUM_BINS_SMOOTH}_smooth.pickle"

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

def main():
    valid_sess = pd.read_pickle(BL_SESSIONS_PATH)
    all_sess_reses = valid_sess.apply(calc_and_save_session, axis=1)
    all_reses = pd.concat(all_sess_reses.values)
    # all_reses.to_pickle("/data/patrick_res/glm_2/belief_state_value_correlations.pickle")
    all_reses.to_pickle("/data/patrick_res/glm_2/bl_belief_state_value_correlations.pickle")

if __name__ == "__main__":
    main()