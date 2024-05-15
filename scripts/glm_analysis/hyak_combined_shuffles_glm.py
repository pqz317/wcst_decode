import utils.behavioral_utils as behavioral_utils
import utils.information_utils as information_utils
import utils.io_utils as io_utils
import pandas as pd
import argparse
from constants.glm_constants import *
import os

OUTPUT_DIR = "/data/res"
MODE = "MeanSubFiringRate"
MODEL = "Linear"

SESSIONS_PATH = "/data/valid_sessions_rpe.pickle"
NUM_SHUFFLES = 1000

def aggregate_shuffles(sess_name, shuffle_type):
    print(f"Processing {sess_name}")
    shuffles = []
    for i in range(NUM_SHUFFLES):
        # shuffle = pd.read_pickle(f"/data/res/{sess_name}_glm_{MODE}_{INTERVAL_SIZE}_{MODEL}_{shuffle_type}_shuffle_{i}.pickle")
        shuffle = pd.read_pickle(os.path.join(OUTPUT_DIR, f"{sess_name}_glm_{EVENT}_{MODE}_{INTERVAL_SIZE}_{MODEL}_maxfeat_shuffle_{i}.pickle"))
        shuffle["ShuffleIdx"] = i
        shuffles.append(shuffle)
    shuffles = pd.concat(shuffles)
    # shuffles.to_pickle(f"/data/res/{sess_name}_glm_{MODE}_{INTERVAL_SIZE}_{MODEL}_{shuffle_type}_shuffles.pickle")
    shuffles.to_pickle(os.path.join(OUTPUT_DIR, f"{sess_name}_glm_{EVENT}_{MODEL}_{INTERVAL_SIZE}_{MODEL}_maxfeat_shuffles.pickle"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle_type', type=str, help="type of shuffle, card or feature_rpe", default="feature_rpe")
    args = parser.parse_args()
    shuffle_type = args.shuffle_type
    print(f"Processing with shuffle type {shuffle_type}")
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    valid_sess.apply(lambda x: aggregate_shuffles(x.session_name, shuffle_type), axis=1)

if __name__ == "__main__":
    main()