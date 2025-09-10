from typing import NamedTuple

LEARNING_RATE = 0.05
MAX_ITER = 500
NUM_TRAIN_PER_COND = 1000
NUM_TEST_PER_COND = 200

P_DROPOUT = 0.5
DECODER_SEED = 42
TEST_RATIO = 0.2

NUM_SPLITS = 8

FB_ONSET_PRE_INTERVAL = 1300
FB_ONSET_POST_INTERVAL = 1500

FB_ONSET_LONG_PRE_INTERVAL = 1800

STIM_ONSET_PRE_INTERVAL = 1000
STIM_ONSET_POST_INTERVAL = 1000

INTERVAL_SIZE = 100

SESS_SPIKES_PATH = "/data/patrick_res/firing_rates/{sub}/{sess_name}_{fr_type}_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"
UNITS_PATH = "/data/patrick_res/firing_rates/{sub}/all_units.pickle"
DRIFT_PATH = "/data/patrick_res/firing_rates/{sub}/drifting_units.pickle"
FEATS_PATH = "/data/patrick_res/sessions/{sub}/feats_at_least_3blocks.pickle"
PAIRS_PATH = "/data/patrick_res/sessions/both/pairs_at_least_3blocks_10sess.pickle"

# TODO: make this backwards compatible
SIG_UNITS_PATH = "/data/patrick_res/firing_rates/{sub}/{event}_{level}_units.pickle"
BOTH_SIG_UNITS_PATH = "/data/patrick_res/firing_rates/{level}_units.pickle"


# regions of interest
REGIONS_OF_INTEREST = ["amygdala_Amy", "basal_ganglia_BG", "inferior_temporal_cortex_ITC", "medial_pallium_MPal", "lateral_prefrontal_cortex_lat_PFC", "anterior_cingulate_gyrus_ACgG"]


MODE_TO_CLASSES = {
    "conf": ["Low", "High"],
    "pref": ["High X", "High Not X"],
    "next_conf": ["Low", "High"],
    "next_pref": ["High X", "High Not X"],
    "feat_belief": ["Low", "High X"],
    "policy": ["X", "Not X"],
    "reward": ["Correct", "Incorrect"],
    "reward_int": ["Correct", "Incorrect"],
    "choice": ["Chose", "Not Chose"],
    "choice_int": ["Chose", "Not Chose"],
    "chose_and_correct": ["Chose Correct", "Not"],
    "updates_beliefs": ["Increases", "Decreases"],
}

MODE_TO_DIRECTION_LABELS = {
    "pref": {"high": "High X", "low": "High Not X"},
    "conf": {"high": "High", "low": "Low"},
    "feat_belief": {"high": "High X", "low": "Low"},
    "next_pref": {"high": "High X", "low": "High Not X"},
    "next_conf": {"high": "High", "low": "Low"},
    "choice_int": {"high": "Chose", "low": "Not Chose"},
    "choice": {"high": "Chose", "low": "Not Chose"},
    "reward": {"high": "Correct", "low": "Incorrect"}
}

MODE_COND_LABEL_MAPS = {
    "conf": None,
    "next_conf": None,
    "pref": None,
    "next_pref": None,
    "feat_belief": None,
    "policy": None,
    "reward": None,
    "choice": None,
    "reward_int": {
        "Chose Correct": "Correct",
        "Chose Incorrect": "Incorrect",
        "Not Chose Correct": "Correct",
        "Not Chose Incorrect": "Incorrect",
    },
    "choice_int": {
        "Chose Correct": "Chose",
        "Chose Incorrect": "Chose",
        "Not Chose Correct": "Not Chose",
        "Not Chose Incorrect": "Not Chose",
    },
    "reward_int": {
        "Chose Correct": "Correct",
        "Chose Incorrect": "Incorrect",
        "Not Chose Correct": "Correct",
        "Not Chose Incorrect": "Incorrect",
    },
    "chose_and_correct": {
        "Chose Correct": "Chose Correct",
        "Chose Incorrect": "Not",
        "Not Chose Correct": "Not",
        "Not Chose Incorrect": "Not",
    },
    "updates_beliefs": {
        "Chose Correct": "Increases",
        "Chose Incorrect": "Decreases",
        "Not Chose Correct": "Decreases",
        "Not Chose Incorrect": "Increases",
    }
}


class TrialInterval(NamedTuple):
    event: str
    pre_interval: int
    post_interval: int
    interval_size: int


def get_trial_interval(trial_event):
    if trial_event == "StimOnset":
        return TrialInterval(
            "StimOnset", 
            STIM_ONSET_PRE_INTERVAL, 
            STIM_ONSET_POST_INTERVAL, 
            INTERVAL_SIZE
        )
    elif trial_event == "FeedbackOnset":
        return TrialInterval(
            "FeedbackOnset", 
            FB_ONSET_PRE_INTERVAL, 
            FB_ONSET_POST_INTERVAL, 
            INTERVAL_SIZE
        )
    elif trial_event == "FeedbackOnsetLong":
        return TrialInterval(
            "FeedbackOnset", 
            FB_ONSET_LONG_PRE_INTERVAL, 
            FB_ONSET_POST_INTERVAL, 
            INTERVAL_SIZE
        )  
    elif trial_event == "decision_warped":
        return TrialInterval(
            "decision_warped",
            0, 600, INTERVAL_SIZE
        )
    else: 
        raise ValueError("unknown trial event")