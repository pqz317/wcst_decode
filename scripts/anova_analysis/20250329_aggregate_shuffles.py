import os
import pandas as pd

from constants.behavioral_constants import *
from constants.decoding_constants import *

import argparse
from anova_configs import add_defaults_to_parser, AnovaConfigs
import utils.io_utils as io_utils
import utils.anova_utils as anova_utils
from tqdm import tqdm

FEATS_PATH = "/data/patrick_res/sessions/{sub}/feats_at_least_3blocks.pickle"
SESSIONS_PATH = "/data/patrick_res/sessions/{sub}/valid_sessions.pickle"

def compute_stats(unit_res, conditions):
    row = {}
    combs = anova_utils.get_combs_of_conds(conditions)
    for comb in combs: 
        combined_cond_str = "".join(comb)
        row[f"{combined_cond_str}_95th"] = unit_res[f"x_{combined_cond_str}_comb_time_fracvar"].quantile(0.95)
        row[f"{combined_cond_str}_99th"] = unit_res[f"x_{combined_cond_str}_comb_time_fracvar"].quantile(0.99)
    # HACK: 
    if "BeliefPartition" in conditions: 
        row["BeliefPref_95th"] = unit_res[f"x_BeliefPref_comb_time_fracvar"].quantile(0.95)
        row["BeliefPref_99th"] = unit_res[f"x_BeliefPref_comb_time_fracvar"].quantile(0.99)
    return pd.Series(row)

def process_feat(args):
    res = []
    for shuffle_idx in tqdm(range(args.num_shuffles)):
        args.shuffle_idx = shuffle_idx
        file_name = io_utils.get_anova_file_name(args)
        output_dir = io_utils.get_anova_output_dir(args)
        df = pd.read_pickle(os.path.join(output_dir, f"{file_name}.pickle"))
        df["shuffle_idx"] = shuffle_idx
        res.append(df)
    res = pd.concat(res)
    # TODO: remove after combined_fracvar is added to anova script
    # combined_cond_str = "".join(args.conditions)
    # res["combined_fracvar"] = res[f"x_{combined_cond_str}_fracvar"] + res[f"x_TimeBins{combined_cond_str}_fracvar"]
    if args.window_size is None:
        stats = res.groupby("PseudoUnitID").apply(lambda x: compute_stats(x, args.conditions)).reset_index()
    else: 
        stats = res.groupby(["PseudoUnitID", "WindowStartMilli"]).apply(lambda x: compute_stats(x, args.conditions)).reset_index()

    # store shuffle stats in parent dir: 
    args.shuffle_idx = None
    output_dir = io_utils.get_anova_output_dir(args)
    stats.to_pickle(os.path.join(output_dir, f"{args.feat}_shuffle_stats.pickle"))

def main():
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(AnovaConfigs(), parser)
    parser.add_argument("--num_shuffles", default=100, type=int)
    args = parser.parse_args()

    args.trial_interval = get_trial_interval(args.trial_event)
    if args.feat_idx is None:
        for feat in FEATURES:
            args.feat = feat
            process_feat(args)
    else: 
        args.feat = FEATURES[args.feat_idx]
        process_feat(args)


if __name__ == "__main__":
    main()