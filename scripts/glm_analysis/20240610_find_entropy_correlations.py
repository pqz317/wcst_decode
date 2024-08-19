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



# # the output directory to store the data
# OUTPUT_DIR = "/data/res"
# SESSIONS_PATH = "/data/valid_sessions_rpe.pickle"
# SPLITS_PATH = "/data/max_feat_kfold_splits.pickle"
# SESS_BEHAVIOR_PATH = "/data/sub-SA_sess-{sess_name}_object_features.csv"
# # path for each session, for spikes that have been pre-aligned to event time and binned. 
# SESS_SPIKES_PATH = "/data/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_{num_bins_smooth}_smooth.pickle"

# # formatting will replace 'feedback_type' first, then replace others in another function
# FEATURE_DIMS = ["Color", "Shape", "Pattern"]

# MIN_BLOCKS_PER_RULE = 2
# NUM_SPLITS = 5

num_bins = 10


def calc_and_save_session(row):
    start = time.time()
    session = row.session_name
    behavior_path = SESS_BEHAVIOR_PATH.format(sess_name=session)
    beh = pd.read_csv(behavior_path)

    # filter trials 
    valid_beh = behavioral_utils.get_valid_trials(beh)
    feature_selections = behavioral_utils.get_selection_features(valid_beh)
    valid_beh = pd.merge(valid_beh, feature_selections, on="TrialNumber", how="inner")
    beh = behavioral_utils.get_feature_values_per_session(session, valid_beh)
    beh = behavioral_utils.get_relative_block_position(beh, num_bins)
    beh = behavioral_utils.get_max_feature_value(beh, num_bins)
    beh = behavioral_utils.calc_feature_probs(beh)
    beh = behavioral_utils.calc_feature_value_entropy(beh, num_bins)

    # beh = behavioral_utils.get_prev_choice_fbs(beh)
    # beh = beh[beh.PrevResponse == "Correct"]

    fr_path = f"/data/patrick_res/firing_rates/{session}_firing_rates_{PRE_INTERVAL}_{EVENT}_{POST_INTERVAL}_{INTERVAL_SIZE}_bins_{NUM_BINS_SMOOTH}_smooth.pickle"
    frs = pd.read_pickle(fr_path)
    agg = frs.groupby(["UnitID", "TrialNumber"]).mean().reset_index()
    reses = []
    for unit in agg.UnitID.unique():
        unit_agg = agg[agg.UnitID == unit]
        merged = pd.merge(unit_agg, beh, on="TrialNumber")
        slope, intercept, r_value, p_value, std_err = stats.linregress(merged["FeatEntropy"], merged["FiringRate"])
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
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    all_sess_reses = valid_sess.apply(calc_and_save_session, axis=1)
    all_reses = pd.concat(all_sess_reses.values)
    all_reses.to_pickle("/data/patrick_res/glm_2/feat_entropy_correlations.pickle")
    # all_reses.to_pickle("/data/patrick_res/glm_2/feat_entropy_correlations_cond_cor.pickle")

if __name__ == "__main__":
    main()