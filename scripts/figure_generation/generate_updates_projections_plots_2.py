"""
Plots for how beliefs predicted from the behavioral model change
as a result of current belief, trial outcomes. 
One plot for conditions: chose X/ cor, chose X, cor, inc, not chose X, chose X /inc
"""
import os
import numpy as np
import pandas as pd
import utils.visualization_utils as visualization_utils

import utils.io_utils as io_utils

import utils.glm_utils as glm_utils
import utils.stats_utils as stats_utils
from matplotlib import pyplot as plt
import matplotlib
import utils.spike_utils as spike_utils
from constants.behavioral_constants import *
from constants.decoding_constants import *
from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import *
import scripts.pseudo_decoding.belief_partitions.belief_partitions_io as belief_partitions_io
from constants.update_projections_constants import *
import seaborn as sns
import itertools


OUTPUT_DIR = "/data/patrick_res/figures/wcst_paper/updates_projections_2_updated"

"""
Try #2 at updates projections, using decoding axes defined from decoding without conditioning 
on choice/reward, also only using time bins prior to cards appear
"""
def format_updates(fig, ax, mode):
    bottom, top = ax.get_ylim()
    max = np.max(np.abs([bottom, top]))
    
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylim([-max, max])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    if mode == "pref":
        ax.set_ylabel(rf"$\Delta \mathbf{{r}}$ along preference of X")
    else: 
        ax.set_ylabel(rf"$\Delta \mathbf{{r}}$ along confidence")
    ax.set_xlabel("")
    visualization_utils.format_plot(ax)
    fig.tight_layout()
    return fig, ax

def get_sig_func(pvals):
    def get_pvals(pair, data1, data2):
        # pairs are always cond, shuffle
        cond, _ = pair
        return pvals[pvals.cond == cond].iloc[0].p
    return get_pvals

def get_sig_func_single(pvals):
    def get_pvals(cond, data):
        return pvals[pvals.cond == cond].iloc[0].p
    return get_pvals

def plot_choice_reward(res, pvals, mode, shuffle_mean):
    order= ["chose X / correct", "correct", "incorrect", "chose X / incorrect"]
    res = res[res.cond.isin(order)]

    res = res.sort_values(by="cond", key=lambda x: x.map(order.index))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(res, x="cond", y="proj", errorbar="se", width=0.5, palette=CONDITION_TO_COLORS, ax=ax)
    ax.axhline(shuffle_mean, color="grey", linestyle="dotted", linewidth=3)

    visualization_utils.add_significance_markers(
        fig, ax, res, "cond", "proj", 
        test=get_sig_func_single(pvals)
    )
    ax.set_xticklabels(["selected X / correct", "correct", "incorrect", "selected X / incorrect"])

    fig, ax = format_updates(fig, ax, mode)

    fig.savefig(f"{OUTPUT_DIR}/{mode}_choice_reward_int.png")
    fig.savefig(f"{OUTPUT_DIR}/{mode}_choice_reward_int.svg")

def plot_chose_reward_by_partitions(res, pvals, mode, reward, shuffle_mean):
    fig, ax = plt.subplots(figsize=(3, 4))
    order= [f"chose X / {reward} / low", f"chose X / {reward} / high X", f"chose X / {reward} / high not X"]
    res = res[res.cond.isin(order)]
    res = res.sort_values(by="cond", key=lambda x: x.map(order.index))
    sns.barplot(res, x="cond", y="proj", errorbar="se", ax=ax, palette=CONDITION_TO_COLORS, order=order)
    ax.axhline(shuffle_mean, color="grey", linestyle="dotted", linewidth=3)

    visualization_utils.add_significance_markers(
        fig, ax, res, "cond", "proj", 
        test=get_sig_func_single(pvals)
    )
    ax.set_xticklabels(["low", "high X", "high not X"])
    fig, ax = format_updates(fig, ax, mode)
    fig.savefig(f"{OUTPUT_DIR}/{mode}_chose_{reward}_by_partition.png", dpi=300)
    fig.savefig(f"{OUTPUT_DIR}/{mode}_chose_{reward}_by_partition.svg")

def main():
    plt.rcParams.update({'font.size': 14})

    args = argparse.Namespace(
        **BeliefPartitionConfigs()._asdict()
    )
    args.subject = "both"
    args.trial_event = TRIAL_EVENT
    args.base_output_path = "/data/patrick_res/update_projections"

    for axis_var in AXIS_VARS:
        args.mode = axis_var
        args.sig_unit_level = f"{args.mode}_99th_no_cond_window_filter_drift"

        print("reading data")
        res = belief_partitions_io.read_update_projections_all_conds(args, CONDITION_MAPS)
        res = res[res.Time <0]
        shuffle_mean = res[res.cond == "shuffle"].proj.mean()
        pvals = belief_partitions_io.read_update_projections_pvals(args, CONDITION_MAPS, axis_vars=[axis_var])
        # print(pvals)

        print("plotting")
        plot_choice_reward(res, pvals, axis_var, shuffle_mean)
        plot_chose_reward_by_partitions(res, pvals, axis_var, "correct", shuffle_mean)
        plot_chose_reward_by_partitions(res, pvals, axis_var, "incorrect", shuffle_mean)


if __name__ == "__main__":
    main()
