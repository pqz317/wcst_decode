
FEATURE_DIMS = ["Color", "Shape", "Pattern"]

FEATURES = [
    'CIRCLE', 'SQUARE', 'STAR', 'TRIANGLE', 
    'CYAN', 'GREEN', 'MAGENTA', 'YELLOW', 
    'ESCHER', 'POLKADOT', 'RIPPLE', 'SWIRL'
]

POSSIBLE_FEATURES = {
    "Color": ['CYAN', 'GREEN', 'MAGENTA', 'YELLOW'],
    "Shape": ['CIRCLE', 'SQUARE', 'STAR', 'TRIANGLE'],
    "Pattern": ['ESCHER', 'POLKADOT', 'RIPPLE', 'SWIRL']
}

FEATURE_TO_DIM = {
    'CIRCLE': 'Shape', 
    'SQUARE': 'Shape', 
    'STAR': 'Shape', 
    'TRIANGLE': 'Shape', 
    'CYAN': 'Color', 
    'GREEN': 'Color', 
    'MAGENTA': 'Color', 
    'YELLOW': 'Color', 
    'ESCHER': 'Pattern', 
    'POLKADOT': 'Pattern', 
    'RIPPLE': 'Pattern', 
    'SWIRL': 'Pattern'
}

BELIEF_PARTITION_THRESH = 0.5

FEEDBACK_TYPES = ["Response", "RPEGroup"]

# SESS_BEHAVIOR_PATH = "/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
# BL_SESS_BEHAVIOR_PATH = "/data/rawdata/sub-BL/sess-{sess_name}/behavior/sub-BL_sess-{sess_name}_object_features.csv"
SESS_BEHAVIOR_PATH = "/data/patrick_res/behavior/{sub}/{sess_name}_object_features.csv"


SESSIONS_PATH = "/data/patrick_res/sessions/{sub}/valid_sessions.pickle"
SA_SESSIONS_PATH = "/data/patrick_res/sessions/SA/valid_sessions_rpe.pickle"
BL_SESSIONS_PATH = "/data/patrick_res/sessions/BL/valid_sessions_61.pickle"
