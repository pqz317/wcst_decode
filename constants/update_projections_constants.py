TRIAL_EVENT = "StimOnset"

REGIONS = [None]

AXIS_VARS = ["pref", "conf"]

CONDITION_MAPS = {
    "chose X / correct": {"Response": "Correct", "Choice": "Chose"},
    "chose X / incorrect": {"Response": "Incorrect", "Choice": "Chose"},
    "correct": {"Response": "Correct"},
    "incorrect": {"Response": "Incorrect"},
    "chose X / incorrect / low": {"Response": "Incorrect", "Choice": "Chose", "BeliefPartition": "Low"},
    "chose X / incorrect / high X": {"Response": "Incorrect", "Choice": "Chose", "BeliefPartition": "High X"},
    "chose X / incorrect / high not X": {"Response": "Incorrect", "Choice": "Chose", "BeliefPartition": "High Not X"},
    "chose X / correct / low": {"Response": "Correct", "Choice": "Chose", "BeliefPartition": "Low"},
    "chose X / correct / high X": {"Response": "Correct", "Choice": "Chose", "BeliefPartition": "High X"},
    "chose X / correct / high not X": {"Response": "Correct", "Choice": "Chose", "BeliefPartition": "High Not X"},
}

FULL_CHOICE_REWARD_COMB_MAPS = {
    "chose X / correct": {"Response": "Correct", "Choice": "Chose"},
    "not chose X / incorrect": {"Response": "Incorrect", "Choice": "Not Chose"},
    "chose": {"Choice": "Chose"},
    "correct": {"Response": "Correct"},
    "incorrect": {"Response": "Incorrect"},
    "not chose": {"Choice": "Not Chose"},
    "not chose X / correct": {"Response": "Correct", "Choice": "Not Chose"},
    "chose X / incorrect": {"Response": "Incorrect", "Choice": "Chose"},
}

CONDITION_TO_COLORS = {
    "chose X / correct": (0.1725, 0.6275, 0.1725, 1.0), # tab green, full alpha
    "chose X / incorrect": (0.8392, 0.1529, 0.1569, 1.0), # tab red, full alpha,
    "correct": (0.586, 0.814, 0.586, 1.0),
    "incorrect": (0.919, 0.576, 0.578, 1.0),
    "chose X / incorrect / low": (0.8392, 0.1529, 0.1569, 1.0),
    "chose X / incorrect / high X": (0.8392, 0.1529, 0.1569, 1.0),
    "chose X / incorrect / high not X": (0.8392, 0.1529, 0.1569, 1.0),
    "chose X / correct / low": (0.1725, 0.6275, 0.1725, 1.0),
    "chose X / correct / high X": (0.1725, 0.6275, 0.1725, 1.0),
    "chose X / correct / high not X": (0.1725, 0.6275, 0.1725, 1.0),
    "shuffle": "grey"
}