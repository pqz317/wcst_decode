from typing import NamedTuple, get_type_hints
import argparse
from distutils.util import strtobool
import json

class AnovaConfigs(NamedTuple):
    # general configs
    conditions: list = []  # specified as comma separated list
    subject: str = "SA"
    feat_idx: int = None
    window_size: int = None
    trial_event: str = "FeedbackOnsetLong"
    time_range: list = None  # specified as comma separated list
    beh_filters: dict = {}  # specified as a json string
    balance_by_filters: bool = False
    fr_type: str = "firing_rates"
    shuffle_method: str = "circular_shift"
    shuffle_idx: int = None
    # >1 makes shuffle_idx the START of a contiguous batch of shuffles run in one process, so a
    # 100-shuffle grid fits in 10 jobs per feature rather than 100. Each shuffle still writes its
    # own pickle under the usual name, so the on-disk layout is identical either way
    num_shuffles_per_job: int = 1
    # skip a shuffle whose pickle already exists. Only meaningful with num_shuffles_per_job > 1,
    # where it lets a preempted job resume rather than redo the shuffles it already finished
    skip_existing: bool = False

    split_idx: int = None

    # label the belief partitions "High X" / "High Not X" rather than "High CIRCLE" / etc, so a
    # --beh_filters value can name a partition without naming a feature. The decoding path already
    # runs this way; the anova path did not, since load_data never passed it through
    use_x: bool = False

    # keep only one half of group B (X chosen, X not preferred), splitting it with
    # stim_belief_vector_alignment.draw_b_split so two runs over B can be given disjoint halves.
    # 1 keeps B1, 2 keeps B2. See claude_notes/stim_belief_single_unit_anova_lite.md part 3
    b_split_half: int = None
    # matches BeliefPartitionConfigs.train_test_seed, so the true run's halves are the same ones
    # the population alignment analysis drew
    split_seed: int = 42

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
            parser.add_argument(f'--{field}', default=value, type=lambda x: [int(t) for t in x.split(",")])
        else: 
            parser.add_argument(f'--{field}', default=value, type=var_type)
    return parser