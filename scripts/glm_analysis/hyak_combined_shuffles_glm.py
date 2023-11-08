import utils.behavioral_utils as behavioral_utils
import utils.information_utils as information_utils
import utils.io_utils as io_utils
import pandas as pd

SESSIONS_PATH = "/data/valid_sessions_rpe.pickle"
NUM_SHUFFLES = 1000

def aggregate_shuffles(sess_name):
    print(f"Processing {sess_name}")
    shuffles = []
    for i in range(NUM_SHUFFLES):
        shuffle = pd.read_pickle(f"/data/res/{sess_name}_glm_card_shuffle_{i}.pickle")
        shuffle["ShuffleIdx"] = i
        shuffles.append(shuffle)
    shuffles = pd.concat(shuffles)
    shuffles.to_pickle(f"/data/res/{sess_name}_glm_card_shuffles.pickle")

def main():
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    valid_sess.apply(lambda x: aggregate_shuffles(x.session_name), axis=1)

if __name__ == "__main__":
    main()