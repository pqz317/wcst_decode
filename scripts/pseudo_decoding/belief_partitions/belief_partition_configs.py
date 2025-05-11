from typing import NamedTuple, get_type_hints
import argparse
from distutils.util import strtobool
import json

class BeliefPartitionConfigs(NamedTuple):
    """
    Set of configurations for performing binary decoding of belief partitions of a single feature
    mode of either confidence (low vs high), preference (high X vs. high not X), or feature belief (low vs. high X)
    """
    mode: str = None  # either conf, pref, or feat_belief
    # general configs
    subject: str = "SA"
    feat_idx: int = None
    pair_idx: int = None
    trial_event: str = "StimOnset"
    beh_filters: dict = {}  # specified as a json string
    balance_by_filters: bool = False
    fr_type: str = "firing_rates"
    # either circular_shift, session_permute, or random
    shuffle_method: str = "session_permute"
    shuffle_idx: int = None
    region_level: str = None
    regions: str = None
    train_test_seed: int = None

    # a file path for loading up significant units
    # a dataframe in pickle format, with feature, PseudoUnitID columns
    sig_unit_level: str = None  

    # decoder configs
    learning_rate: float = 0.05
    max_iter: int = 500
    num_train_per_cond: int = 1000
    num_test_per_cond: int = 200
    p_dropout: float = 0.5
    test_ratio: float = 0.2
    num_splits: int = 8
    use_v2_pseudo: bool = True

    # file storage, naming
    run_name: str = None
    base_output_path: str = "/data/patrick_res/belief_partitions"



def add_defaults_to_parser(default_configs, parser):
    # Automatically add arguments based on the namedtuple fields
    for field, value in default_configs._asdict().items():
        # print(default_configs.__annotations__)
        var_type = get_type_hints(default_configs)[field]
        if var_type is bool: 
            parser.add_argument(f'--{field}', default=value, type=lambda x: bool(strtobool(x)))
        elif field == "beh_filters": 
            parser.add_argument(f'--{field}', default=value, type=lambda x: json.loads(x))
        else: 
            parser.add_argument(f'--{field}', default=value, type=var_type)
    return parser