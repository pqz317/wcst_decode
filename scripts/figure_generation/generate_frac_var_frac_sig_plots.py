"""
Plots for a fraction of variance explained a number of signficant units for each variable
For each variable, metric, make two plots: 
one for the whole population, one for regions split up
"""
import os
import numpy as np
import pandas as pd
import utils.visualization_utils as visualization_utils
import utils.anova_utils as anova_utils
import utils.spike_utils as spike_utils

import utils.io_utils as io_utils

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

MODE_TO_COND = {
    "pref": "BeliefPref",
    "conf": "BeliefConf",
    "choice": "Choice",
    "reward": "Response"
}
MODES = ["choice", "reward", "pref", "conf"]
# MODES = ["choice"]
REGIONS = ["amygdala_Amy", "basal_ganglia_BG", "inferior_temporal_cortex_ITC", "medial_pallium_MPal", "lateral_prefrontal_cortex_lat_PFC", "anterior_cingulate_gyrus_ACgG"]
# REGIONS = ["amygdala_Amy"]
OUTPUT_DIR = "/data/patrick_res/figures/wcst_paper/frac_var_frac_sig_updated"


def plot(res, y_col, hue_col=None, hue_order=None, palette=None, hue_display_names=None, color=None):
    y_col_to_label = {
        "frac_units_sig": "% of units significant",
        "frac_variance": "% of variance explained",
    }
    if hue_display_names:
        print(res[hue_col].unique())
        res[hue_col] = res[hue_col].map(hue_display_names)
    res["Time"] = res.WindowEndMilli / 1000 - 0.25 # the middle of the 500ms window 
    res[y_col] = res[y_col] * 100 # plot as percentages
    fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharey='row', width_ratios=[20, 33])
    for i, event in enumerate(["StimOnset", "FeedbackOnsetLong"]):
        trial_res = res[res.trial_event == event]
        sns.lineplot(
            trial_res, x="Time", y=y_col, 
            hue=hue_col, hue_order=hue_order, palette=palette, color=color,
            linewidth=3, ax=axs[i], errorbar="se"
        )

    ax1, ax2 = axs
    ax1.axvline(-.5, color='grey', linestyle='dotted', linewidth=3)
    ax1.axvline(0, color='grey', linestyle='dotted', linewidth=3)
    ax2.axvline(-.8, color='grey', linestyle='dotted', linewidth=3)
    ax2.axvline(0, color='grey', linestyle='dotted', linewidth=3)
    ax1.set_ylabel(y_col_to_label[y_col])
    ax1.set_xlabel(f"Time to cards appear (s)")
    stim_ticks = [-1, -.5, 0, .5, 1]
    ax1.set_xticks(stim_ticks)
    ax1.set_xticklabels(stim_ticks)

    ax2.set_xlabel(f"Time to feedback (s)")
    fb_ticks = [-1.5, -1, -.5, 0, .5, 1, 1.5]
    ax2.set_xticks(fb_ticks)
    ax2.set_xticklabels(fb_ticks)
    if hue_col is not None: 
        ax2.get_legend().remove()
        for line in ax1.legend().get_lines():
            line.set_linewidth(6)
    visualization_utils.format_plot(axs)
    fig.tight_layout()
    return fig, axs


def plot_for_mode(mode, by_region):
    args = argparse.Namespace(
        **AnovaConfigs()._asdict()
    )
    if mode in ["choice", "reward"]:
        args.conditions = ["Choice", "Response"]
        args.beh_filters = {}
    else: 
        args.conditions = ["BeliefConf", "BeliefPartition"]
        args.beh_filters = {"Response": "Correct", "Choice": "Chose"}
    args.window_size = 500
    args.sig_thresh = "99th"

    if not by_region:
        color = visualization_utils.MODE_TO_COLOR[visualization_utils.MODE_TO_DISPLAY_NAMES[mode]]
        frac_sig = anova_utils.num_sig_units_by_time(args, [MODE_TO_COND[mode]], sig_thresh="99th")
        fig, axs = plot(frac_sig, "frac_units_sig", color=color)
        fig.savefig(f"{OUTPUT_DIR}/{mode}_frac_sig_whole_pop.png")
        fig.savefig(f"{OUTPUT_DIR}/{mode}_frac_sig_whole_pop.svg")
        plt.close(fig)        

        frac_var = anova_utils.frac_var_explained_by_time(args, [MODE_TO_COND[mode]])
        fig, axs = plot(frac_var, "frac_variance", color=color)
        fig.savefig(f"{OUTPUT_DIR}/{mode}_frac_var_whole_pop.png")
        fig.savefig(f"{OUTPUT_DIR}/{mode}_frac_var_whole_pop.svg")
    else: 
        sig_res = []
        var_res = []
        for region in REGIONS:
            region_units = spike_utils.get_all_region_units("structure_level2_cleaned", region)
            frac_sig = anova_utils.num_sig_units_by_time(args, [MODE_TO_COND[mode]], sig_thresh="99th", units=region_units)
            frac_sig["region"] = region
            frac_var = anova_utils.frac_var_explained_by_time(args, [MODE_TO_COND[mode]], units=region_units)
            frac_var["region"] = region

            sig_res.append(frac_sig)
            var_res.append(frac_var)
        sig_res = pd.concat(sig_res)
        var_res = pd.concat(var_res)
        fig, axs = plot(
            sig_res, "frac_units_sig", 
            hue_col="region",
            hue_order=visualization_utils.REGION_ORDER, 
            palette=visualization_utils.REGION_TO_COLOR,
            hue_display_names=visualization_utils.REGION_TO_DISPLAY_NAMES
        )
        fig.savefig(f"{OUTPUT_DIR}/{mode}_frac_sig_by_region.png")
        fig.savefig(f"{OUTPUT_DIR}/{mode}_frac_sig_by_region.svg")
        plt.close(fig)        

        fig, axs = plot(
            var_res, "frac_variance",
            hue_col="region",
            hue_order=visualization_utils.REGION_ORDER, 
            palette=visualization_utils.REGION_TO_COLOR,
            hue_display_names=visualization_utils.REGION_TO_DISPLAY_NAMES
        )
        fig.savefig(f"{OUTPUT_DIR}/{mode}_frac_var_by_region.png")
        fig.savefig(f"{OUTPUT_DIR}/{mode}_frac_var_by_region.svg")



def main():
    plt.rcParams.update({'font.size': 14})
    for mode in tqdm(MODES):
        plot_for_mode(mode, by_region=False)
        plot_for_mode(mode, by_region=True)

if __name__ == "__main__":
    main()


