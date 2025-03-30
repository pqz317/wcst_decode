from typing import NamedTuple, get_type_hints
import argparse
from distutils.util import strtobool
import json

class AnovaConfigs(NamedTuple):
    # general configs
    conditions: list = []  # specified as comma separated list
    subject: str = "SA"
    feat_idx: int = None
    trial_event: str = "FeedbackOnsetLong"
    time_range: list = None  # specified as comma separated list
    beh_filters: dict = {}  # specified as a json string
    balance_by_filters: bool = False
    fr_type: str = "firing_rates"
    shuffle_method: str = "circular_shift"
    shuffle_idx: int = None

    # file storage, naming
    run_name: str = None
    base_output_path: str = "/data/patrick_res/anova"


def add_defaults_to_parser(default_configs, parser):
    # Automatically add arguments based on the namedtuple fields
    for field, value in default_configs._asdict().items():
        # print(default_configs.__annotations__)
        var_type = get_type_hints(default_configs)[field]
        if var_type is bool: 
            parser.add_argument(f'--{field}', default=value, type=lambda x: bool(strtobool(x)))
        elif field == "beh_filters": 
            parser.add_argument(f'--{field}', default=value, type=lambda x: json.loads(x))
        elif field == "conditions":
            parser.add_argument(f'--{field}', default=value, type=lambda x: x.split(","))
        elif field == "time_range":
            parser.add_argument(f'--{field}', default=value, type=lambda x: x.split(","))
        else: 
            parser.add_argument(f'--{field}', default=value, type=var_type)
    return parser