from typing import NamedTuple, get_type_hints
import argparse
from distutils.util import strtobool
import json

class SingleSelectedFeatureConfigs(NamedTuple):
    condition: str = "chosen"  # either chosen, pref, or not_pref, pref vs not pref
    # general configs
    subject: str = "SA"
    feat_idx: int = None
    trial_event: str = "StimOnset"
    beh_filters: dict = {}  # specified as a json string
    balance_by_filters: bool = False
    fr_type: str = "firing_rates"
    shuffle_idx: int = None
    region_level: str = None
    regions: str = None
    train_test_seed: int = None

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
    base_output_path: str = "/data/patrick_res/single_selected_feature"


class SingleSelectedFeatureCrossCondConfigs(NamedTuple):
    condition: str = None
    model_cond: str = None
    data_cond: str = None
    # general configs
    subject: str = "SA"
    feat_idx: int = None
    trial_event: str = "StimOnset"
    beh_filters: dict = {}  # specified as a json string
    balance_by_filters: bool = False
    fr_type: str = "firing_rates"
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
    use_v2_pseudo: bool = True

    # file storage, naming
    run_name: str = None
    base_output_path: str = "/data/patrick_res/single_selected_feature"



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