"""
For each region, for each variable, generate some single unit psths, along with rasters
"""
import os
import numpy as np
import pandas as pd
import utils.behavioral_utils as behavioral_utils
import utils.information_utils as information_utils
import utils.visualization_utils as visualization_utils
import utils.pseudo_classifier_utils as pseudo_classifier_utils
import utils.classifier_utils as classifier_utils

import utils.io_utils as io_utils
import utils.spike_utils as spike_utils

import utils.glm_utils as glm_utils
from matplotlib import pyplot as plt
import matplotlib
from constants.behavioral_constants import *
from constants.decoding_constants import *
import seaborn as sns
from scripts.anova_analysis.anova_configs import *
from scripts.anova_analysis.run_anova import load_data
import scipy
import argparse
import copy
import itertools
from tqdm import tqdm

SUBJECTS = ["SA", "BL"]

MODE_TO_COND = {
    "pref": "BeliefPref",
    "conf": "BeliefConf",
    "choice": "Choice",
    "reward": "Response"
}
MODES = ["choice", "reward", "pref", "conf"]
REGIONS = ["amygdala_Amy", "basal_ganglia_BG", "inferior_temporal_cortex_ITC", "medial_pallium_MPal", "lateral_prefrontal_cortex_lat_PFC", "anterior_cingulate_gyrus_ACgG"]

# MODES = ["pref"]
# REGIONS = ["anterior_cingulate_gyrus_ACgG"]

def plot_for_mode_region(subject, mode, region):
    args = argparse.Namespace(
        **AnovaConfigs()._asdict()
    )
    if mode in ["choice", "reward"]:
        args.conditions = ["Choice", "Response"]
        args.beh_filters = {}
    else: 
        args.conditions = ["BeliefConf", "BeliefPartition"]
        args.beh_filters = {"Response": "Correct", "Choice": "Chose"}
    args.subject = subject
    args.window_size = 500
    args.sig_thresh = "99th"

    valid_units = spike_utils.get_subject_units(subject)
    valid_units = spike_utils.filter_drift(valid_units, subject)

    all_good_units = []
    res = []
    for trial_event in ["StimOnset", "FeedbackOnsetLong"]:
        args.trial_event = trial_event
        event_res = io_utils.read_anova_good_units(args, args.sig_thresh, MODE_TO_COND[mode], return_pos=True)
        region_res = event_res[event_res.structure_level2_cleaned == region]
        # used to grab p values
        res.append(region_res)
        col_name = f"x_{MODE_TO_COND[mode]}_comb_time_fracvar"
        summed = region_res.groupby(["PseudoUnitID", "feat"])[col_name].sum().reset_index()
        region_top = summed.sort_values(col_name, ascending=False).drop_duplicates('PseudoUnitID').head(5)
        all_good_units.append(region_top)
    all_good_units = pd.concat(all_good_units).drop_duplicates("PseudoUnitID")
    # make sure the units found are valid, not drifing. s
    all_good_units = all_good_units[all_good_units.PseudoUnitID.isin(valid_units.PseudoUnitID)]

    res = pd.concat(res)
    # set significance times to be middle of the window
    res["Time"] = res.WindowEndMilli / 1000 - 0.25 # the middle of the 500ms window 

    for i, row in all_good_units.iterrows():
        fig, axs = visualization_utils.plot_psth_both_events(mode, int(row.PseudoUnitID), row.feat, args, pval_res=res)
        fig.savefig(f"/data/patrick_res/figures/wcst_paper/single_unit_psths_filter_drift/{mode}_{region}_{subject}_{row.PseudoUnitID}_{row.feat}.png", dpi=300)
        fig.savefig(f"/data/patrick_res/figures/wcst_paper/single_unit_psths_filter_drift/{mode}_{region}_{subject}_{row.PseudoUnitID}_{row.feat}.svg")
        plt.close(fig)

def main():
    plt.rcParams.update({'font.size': 14})
    parser = argparse.ArgumentParser()
    parser.add_argument(f'--run_id', default=None, type=int)
    args = parser.parse_args()

    

    mode_regions = list(itertools.product(SUBJECTS, MODES, REGIONS))
    if args.run_id is not None: 
        subject, mode, region = mode_regions[args.run_id]
        plot_for_mode_region(subject, mode, region)
    else:         
        for (subject, mode, region) in tqdm(mode_regions):
            plot_for_mode_region(subject, mode, region)

if __name__ == "__main__":
    main()