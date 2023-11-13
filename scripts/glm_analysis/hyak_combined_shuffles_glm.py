import utils.behavioral_utils as behavioral_utils
import utils.information_utils as information_utils
import utils.io_utils as io_utils
import pandas as pd
import argparse

SESSIONS_PATH = "/data/valid_sessions_rpe.pickle"
NUM_SHUFFLES = 1000

def aggregate_shuffles(sess_name, shuffle_type):
    print(f"Processing {sess_name}")
    shuffles = []
    for i in range(NUM_SHUFFLES):
        shuffle = pd.read_pickle(f"/data/res/{sess_name}_glm_{shuffle_type}_shuffle_{i}.pickle")
        shuffle["ShuffleIdx"] = i
        shuffles.append(shuffle)
    shuffles = pd.concat(shuffles)
    shuffles.to_pickle(f"/data/res/{sess_name}_glm_{shuffle_type}_shuffles.pickle")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle_type', type=str, help="type of shuffle, card or feature_rpe", default="feature_rpe")
    args = parser.parse_args()
    shuffle_type = args.shuffle_type

    valid_sess = pd.read_pickle(SESSIONS_PATH)
    valid_sess.apply(lambda x: aggregate_shuffles(x.session_name, shuffle_type), axis=1)

if __name__ == "__main__":
    main()