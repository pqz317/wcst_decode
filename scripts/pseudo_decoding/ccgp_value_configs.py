from typing import NamedTuple
import argparse
from distutils.util import strtobool


class CCGPValueConfigs(NamedTuple):
    # general configs
    subject: str = "SA"
    pair_idx: int = None
    trial_event: str = "StimOnset"
    use_next_trial_value: bool = False
    fr_type: str = "firing_rates"
    prev_response: str = None
    shuffle_idx: int = None
    region_level: str = None
    regions: str = None

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
    base_output_path: str = "/data/patrick_res/ccgp_value"



def add_defaults_to_parser(parser):
    # Automatically add arguments based on the namedtuple fields
    default_configs = CCGPValueConfigs()
    for field, value in default_configs._asdict().items():
        var_type = default_configs. __annotations__[field]
        if var_type is bool: 
            parser.add_argument(f'--{field}', default=value, type=lambda x: bool(strtobool(x)))
        else: 
            parser.add_argument(f'--{field}', default=value, type=var_type)
    return parser