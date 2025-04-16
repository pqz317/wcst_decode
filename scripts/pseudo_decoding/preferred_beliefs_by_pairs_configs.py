from typing import NamedTuple, get_type_hints
import argparse
from distutils.util import strtobool
import json


class PreferredBeliefsByPairsConfigs(NamedTuple):
    # general configs
    subject: str = "SA"
    pair_idx: int = None
    trial_event: str = "StimOnset"
    beh_filters: dict = {}  # specified as a json string
    balance_by_filters: bool = False
    chosen_not_preferred: bool = False
    high_val_only: bool = False
    fr_type: str = "firing_rates"
    shuffle_idx: int = None
    region_level: str = None
    regions: str = None

    # a file path for loading up significant units
    # a dataframe in pickle format, with feature, PseudoUnitID columns
    sig_unit_level: str = None  
    shuffle_method: str = "circular_shift"

    # decoder configs
    learning_rate: float = 0.05
    max_iter: int = 500
    num_train_per_cond: int = 1000
    num_test_per_cond: int = 200
    p_dropout: float = 0.5
    test_ratio: float = 0.2
    num_splits: int = 8

    # file storage, naming
    run_name: str = None
    base_output_path: str = "/data/patrick_res/preferred_beliefs"


def add_defaults_to_parser(parser):
    # Automatically add arguments based on the namedtuple fields
    default_configs = PreferredBeliefsByPairsConfigs()
    for field, value in default_configs._asdict().items():
        var_type = get_type_hints(default_configs)[field]
        if var_type is bool: 
            parser.add_argument(f'--{field}', default=value, type=lambda x: bool(strtobool(x)))
        elif field == "beh_filters": 
            parser.add_argument(f'--{field}', default=value, type=lambda x: json.loads(x))
        else: 
            parser.add_argument(f'--{field}', default=value, type=var_type)
    return parser