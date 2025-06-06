import numpy as np
import pandas as pd
from spike_tools import (
    general as spike_general,
    analysis as spike_analysis,
)
import utils.behavioral_utils as behavioral_utils
import utils.spike_utils as spike_utils
import utils.io_utils as io_utils
import os
from distutils.util import strtobool
from constants.behavioral_constants import *
import argparse
from constants.decoding_constants import *
from scripts.anova_analysis.anova_configs import add_defaults_to_parser, AnovaConfigs


"""
Per feature, generates lists of significant units based off of some ANOVA signficance:
- Either belief partition or conf, pref separately
- Either by event, or sliding window
- Either 95th or 99th percentile
- Either include drifing units or not
For each option, report number of units this would give per region/feature. 
- sig units/num sessions for SA,BL individually, then for both
"""

DRIFT_PATH = "/data/patrick_res/firing_rates/{sub}/drifting_units.pickle"
REGIONS = ["inferior_temporal_cortex (ITC)", "medial_pallium (MPal)", "basal_ganglia (BG)", "amygdala (Amy)"]


def load_anova_res_for_event(args):
    if args.sig_type == "pref_conf":
        pref_res = io_utils.read_anova_good_units(args, args.sig_thresh, "BeliefPref", return_pos=True)
        conf_res = io_utils.read_anova_good_units(args, args.sig_thresh, "BeliefConf", return_pos=True)
        return pd.concat([pref_res, conf_res])
    elif args.sig_type == "belief_partition":
        return io_utils.read_anova_good_units(args, args.sig_thresh, "BeliefPartition", return_pos=True)


def find_sig_units_for_sub(args):
    event_reses = []
    for event in ["StimOnset", "FeedbackOnsetLong"]:
        args.trial_event = event
        event_reses.append(load_anova_res_for_event(args))
    res = pd.concat(event_reses)
    res = res.groupby(["feat", "structure_level2"]).PseudoUnitID.unique().reset_index().explode("PseudoUnitID")
    res["session"] = (res.PseudoUnitID / 100).astype(int)
    return res

def filter_drift(units, args):
    drift_units = pd.read_pickle(DRIFT_PATH.format(sub=args.subject))
    return units[~units.PseudoUnitID.isin(drift_units.PseudoUnitID)]

def get_sub_stats(units, region):
    units = units[units.structure_level2 == region]
    per_feat_sig = units.groupby("feat").PseudoUnitID.nunique().reindex(FEATURES, fill_value=0).reset_index(name="num_units_sig")
    per_feat_sess = units.groupby("feat").session.nunique().reindex(FEATURES, fill_value=0).reset_index(name="num_sessions_sig")

    res = pd.merge(per_feat_sig, per_feat_sess, on="feat", how="outer")
    res = res.fillna(0)
    return res

def report_region_stats(all_units):
    sa_units, bl_units = all_units
    for region in REGIONS:
        sa_stats = get_sub_stats(sa_units, region)
        bl_stats = get_sub_stats(bl_units, region)
        stats = pd.merge(sa_stats, bl_stats, on="feat", how="outer", suffixes=["_sa", "_bl"])
        stats["num_units_sig_total"] = stats["num_units_sig_sa"] + stats["num_units_sig_bl"]
        stats["num_sessions_sig_total"] = stats["num_sessions_sig_sa"] + stats["num_sessions_sig_bl"]
        print(region)
        print(stats.to_string())

def get_sig_level_str(args):
    window_str = "window" if args.window_size else None
    filter_str = "filter_drift" if args.filter_drift else None
    parts = [args.sig_type, args.sig_thresh, window_str, filter_str]
    return "_".join(x for x in parts if x)

def main():
    parser = argparse.ArgumentParser()
    # just use ANOVA configs, and add a few. 
    parser = add_defaults_to_parser(AnovaConfigs(), parser)
    parser.add_argument('--sig_type', default="pref_conf", type=str)
    parser.add_argument('--sig_thresh', default="95th", type=str)
    parser.add_argument('--filter_drift', default=True, type=lambda x: bool(strtobool(x)))
    parser.add_argument('--dry_run', default=True, type=lambda x: bool(strtobool(x)))

    args = parser.parse_args()

    # These should just be set always
    args.conditions = ["BeliefConf", "BeliefPartition"]
    args.beh_filters = {"Response": "Correct", "Choice": "Chose"}

    all_units = []
    for sub in ["SA", "BL"]:
        args.subject = sub
        sub_units = find_sig_units_for_sub(args)
        if args.filter_drift:
            sub_units = filter_drift(sub_units, args)

        sig_level_str = get_sig_level_str(args)
        print(f"Saving units with sig level {sig_level_str}")
        if not args.dry_run:
            # SIG_UNITS_PATH = "/data/patrick_res/firing_rates/{sub}/{event}_{level}_units.pickle"
            sub_units.to_pickle(f"/data/patrick_res/firing_rates/{sub}/FeedbackOnsetLong_{sig_level_str}_units.pickle")
            sub_units.to_pickle(f"/data/patrick_res/firing_rates/{sub}/StimOnset_{sig_level_str}_units.pickle")
        all_units.append(sub_units)
    report_region_stats(all_units)
    

if __name__ == "__main__":
    main()



