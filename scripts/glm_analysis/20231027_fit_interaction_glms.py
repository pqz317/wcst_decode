import os
import numpy as np
import pandas as pd
import utils.behavioral_utils as behavioral_utils
import utils.information_utils as information_utils
from matplotlib import pyplot as plt
from multiprocessing import Pool
import time
from sklearn.linear_model import PoissonRegressor


EVENT = "FeedbackOnset"  # event in behavior to align on
PRE_INTERVAL = 1300   # time in ms before event
POST_INTERVAL = 1500  # time in ms after event
INTERVAL_SIZE = 100  # size of interval in ms
NUM_BINS_SMOOTH = 1

# the output directory to store the data
OUTPUT_DIR = "/data/patrick_res/information"
SESSIONS_PATH = "/data/patrick_res/multi_sess/valid_sessions_rpe.pickle"

# path for each session, specifying behavior
SESS_BEHAVIOR_PATH = "/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/patrick_res/multi_sess/{sess_name}/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_{num_bins_smooth}_smooth.pickle"

DATA_MODE = "SpikeCounts"

NUM_SHUFFLES = 1000
NUM_PROCESSES = 30

SEED = 42

FEATURE_DIMS = ["Color", "Shape", "Pattern"]
INTERACTIONS = [f"{dim}RPE" for dim in FEATURE_DIMS]


def load_data(row):
    sess_name = row.session_name
    behavior_path = SESS_BEHAVIOR_PATH.format(sess_name=sess_name)
    beh = pd.read_csv(behavior_path)

    # filter trials 
    valid_beh = behavioral_utils.get_valid_trials(beh)
    feature_selections = behavioral_utils.get_selection_features(valid_beh)
    valid_beh = pd.merge(valid_beh, feature_selections, on="TrialNumber")

    # grab the features of the selected card
    valid_beh_rpes = behavioral_utils.get_rpes_per_session(sess_name, valid_beh)
    assert len(valid_beh) == len(valid_beh_rpes)
    pos_med = row.FE_pos_median
    neg_med = row.FE_neg_median
    # add median labels to 
    def add_group(row):
        rpe = row.RPE_FE
        group = None
        if rpe < neg_med:
            group = "more neg"
        elif rpe >= neg_med and rpe < 0:
            group = "less neg"
        elif rpe >= 0 and rpe < pos_med:
            group = "less pos"
        elif rpe > pos_med:
            group = "more pos"
        row["RPEGroup"] = group
        return row
    valid_beh_rpes = valid_beh_rpes.apply(add_group, axis=1)
    for feature_dim in FEATURE_DIMS:
        valid_beh_rpes[f"{feature_dim}RPE"] = valid_beh_rpes[feature_dim] + "_" + valid_beh_rpes["RPEGroup"]

    valid_beh_rpes = valid_beh_rpes.set_index(["TrialNumber"])

    spikes_path = SESS_SPIKES_PATH.format(
        sess_name=sess_name, 
        pre_interval=PRE_INTERVAL, 
        event=EVENT, 
        post_interval=POST_INTERVAL, 
        interval_size=INTERVAL_SIZE,
        num_bins_smooth=NUM_BINS_SMOOTH,
    )
    frs = pd.read_pickle(spikes_path)
    frs = frs.set_index(["TrialNumber"])
    return valid_beh_rpes, frs

def flatten_columns(beh, columns):
    flattened_columns = []
    for column in columns:
        values = beh[column].unique()
        for value in values:
            beh[value] = (beh[column] == value).astype(int)
        flattened_columns.extend(values)
    return beh, flattened_columns

def fit_glm(df, x_cols, y_col):
    ys = df[y_col].values
    xs = df[x_cols].values
    model = PoissonRegressor(alpha=1)
    model = model.fit(xs, ys)
    return pd.Series({"score": model.score(xs, ys)})

def fit_glms_by_unit_and_time(data, x_inputs):
    data, flattened_columns = flatten_columns(data, x_inputs)
    res = data.groupby(["UnitID", "TimeBins"]).apply(lambda x: fit_glm(x, flattened_columns, "SpikeCounts")).reset_index()
    # res = data.groupby("TimeBins").apply(lambda x: fit_glm_torch(x, flattened_columns, "SpikeCounts")).reset_index()
    return res.fillna(0)

def fit_glm_for_data(data, input_columns):
    beh, frs = data
    beh_inputs = beh[input_columns]
    data = pd.merge(beh_inputs, frs, on="TrialNumber")
    res = fit_glms_by_unit_and_time(data, input_columns)
    return res

def create_shuffles(data, columns, rng):
    for column in columns:
        vals = data[column].values
        rng.shuffle(vals)
        data[column] = vals
    return data

def calc_for_shuffle(args):
    i, row = args
    print(f"Calculating shuffle #{i}")
    beh, frs = load_data(row)
    rng = np.random.default_rng()
    input_columns = ["RPEGroup"] + FEATURE_DIMS + INTERACTIONS

    beh_inputs_to_shuffle = beh[input_columns]
    shuffle_columns = INTERACTIONS
    shuffled_beh = create_shuffles(beh_inputs_to_shuffle, shuffle_columns, rng)
    # shuffled_data = pd.merge(shuffled_inputs, frs, on="TrialNumber")
    # shuffled_res = fit_glms_by_time(shuffled_data, input_columns)
    shuffled_res = fit_glm_for_data((shuffled_beh, frs), input_columns)
    shuffled_res["ShuffledIdx"] = i
    return shuffled_res

def create_interaction_shuffles(row):
    args = [(i, row) for i in range(NUM_SHUFFLES)]
    with Pool(processes=NUM_PROCESSES) as pool:
        res = pool.map(calc_for_shuffle, args)
    shuffled_mis = pd.concat(res)
    return shuffled_mis

    
def calc_and_save_session(row):
    session = row.session_name
    start = time.time()
    print(f"Processing session {session}")
    data = load_data(row)
    separate_input_cols = ["RPEGroup"] + FEATURE_DIMS
    separate_reses = fit_glm_for_data(data, separate_input_cols)
    separate_reses.to_pickle(os.path.join(OUTPUT_DIR, f"{session}_glm_feature_rpe_separate.pickle"))

    interaction_input_cols = ["RPEGroup"] + FEATURE_DIMS + INTERACTIONS
    interaction_reses = fit_glm_for_data(data, interaction_input_cols)
    interaction_reses.to_pickle(os.path.join(OUTPUT_DIR, f"{session}_glm_feature_rpe_interaction.pickle"))

    print(f"Separate and interaction took {(time.time() - start) / 60} minutes, starting shuffles")

    shuffled_reses = create_interaction_shuffles(row)
    shuffled_reses.to_pickle(os.path.join(OUTPUT_DIR, f"{session}_glm_feature_rpe_interaction_shuffled.pickle"))

    end = time.time()
    print(f"Session {session} took {(end - start) / 60} minutes")


def main():
    start = time.time()
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    # TODO: remove next line
    valid_sess = valid_sess[valid_sess.session_name == "20180802"]
    valid_sess.apply(calc_and_save_session, axis=1)
    end = time.time()
    print(f"Whole script took {(end - start) / 60} minutes")


if __name__ == "__main__":
    main()

