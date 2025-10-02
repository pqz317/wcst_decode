"""
Plots for how beliefs predicted from the behavioral model change
as a result of current belief, trial outcomes. 
One plot for conditions: chose X/ cor, chose X, cor, inc, not chose X, chose X /inc
"""
import os
import numpy as np
import pandas as pd
import utils.behavioral_utils as behavioral_utils
import utils.visualization_utils as visualization_utils
import utils.stats_utils as stats_utils

from matplotlib import pyplot as plt
from constants.behavioral_constants import *
from constants.decoding_constants import *
from constants.update_projections_constants import *
import seaborn as sns




"""
TODO: for now, stop thinking about shuffles and just get some plots plotted, will circle back... 
Also, just plot for SA first, worry about combining subjects later

For all behavior (should I combine bewteen both subjects? maybe try this first)
load all behavior,  
how should I do these shuffles... 
load true beh per session, and then load
"""
def format_updates(fig, ax):
    bottom, top = ax.get_ylim()
    ax.axhline(0, color="black", linewidth=1)
    max = np.max(np.abs([bottom, top]))
    ax.set_ylim([-max, max])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_xlabel("")
    ax.set_ylabel(r"$\Delta b(X)$")

    visualization_utils.format_plot(ax)
    # overwriting some default visualizations
    # ax.spines["bottom"].set_visible(False)   # remove axis line
    # ax.tick_params(axis="x", length=0) 
    fig.tight_layout()
    return fig, ax

def plot_choice_reward(stats_res, subject, shuffle_vals):
    order= ["chose X / correct", "correct", "incorrect", "chose X / incorrect"]
    stats_res = stats_res[stats_res.cond.isin(order)]

    stats_res = stats_res.sort_values(by="cond", key=lambda x: x.map(order.index))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(stats_res, x="cond", y="change in prob", errorbar="se", width=0.5, palette=CONDITION_TO_COLORS, ax=ax)
    ax.axhline(shuffle_vals.mean(), color="grey", linestyle="dotted", linewidth=3)

    # visualization_utils.add_significance_bars(fig, ax, stats_res, "cond", "change in prob", pairs=[
    #     ("chose X / correct", "shuffle"),
    #     ("chose X / incorrect", "shuffle"),
    #     ("correct", "shuffle"),
    #     ("incorrect", "shuffle"),
    # ], 
    # # test=stats_utils.get_permutation_test_func(test_type="two_side")
    # )

    visualization_utils.add_significance_markers(
        fig, ax, stats_res, "cond", "change in prob", 
        test=stats_utils.get_permutation_test_func_single(shuffle_vals, test_type="two_side")
    )
    ax.set_xticklabels(["selected X / correct", "correct", "incorrect", "selected X / incorrect"])

    fig, ax = format_updates(fig, ax)

    fig.savefig(f"/data/patrick_res/figures/wcst_paper/change_in_beliefs_by_obs/{subject}_choice_reward_int.png")
    fig.savefig(f"/data/patrick_res/figures/wcst_paper/change_in_beliefs_by_obs/{subject}_choice_reward_int.svg")

def plot_chose_reward_by_partitions(stats_res, reward, subject, shuffle_vals):
    fig, ax = plt.subplots(figsize=(3, 4))
    order= [f"chose X / {reward} / low", f"chose X / {reward} / high X", f"chose X / {reward} / high not X"]
    stats_res = stats_res[stats_res.cond.isin(order)]
    stats_res = stats_res.sort_values(by="cond", key=lambda x: x.map(order.index))
    sns.barplot(stats_res, x="cond", y="change in prob", errorbar="se", palette=CONDITION_TO_COLORS, ax=ax, order=order)
    ax.axhline(shuffle_vals.mean(), color="grey", linestyle="dotted", linewidth=3)

    visualization_utils.add_significance_markers(
        fig, ax, stats_res, "cond", "change in prob", 
        test=stats_utils.get_permutation_test_func_single(shuffle_vals, test_type="two_side")
    )

    ax.set_xticklabels(["low", "high X", "high not X"])

    fig, ax = format_updates(fig, ax)
    fig.savefig(f"/data/patrick_res/figures/wcst_paper/change_in_beliefs_by_obs/{subject}_chose_{reward}_by_partition.png", dpi=300)
    fig.savefig(f"/data/patrick_res/figures/wcst_paper/change_in_beliefs_by_obs/{subject}_chose_{reward}_by_partition.svg")


def main():
    plt.rcParams.update({'font.size': 14})
    subject = "BL"
    all_beh = behavioral_utils.load_all_beh_for_sub(subject)
    feat_sessions = pd.read_pickle(FEATS_PATH.format(sub=subject))
    stats_res = []
    for i, row in feat_sessions.iterrows():
        for session in row.sessions:
            sess_beh = all_beh[all_beh.session == session].copy()
            res = behavioral_utils.get_belief_changes_by_obs(sess_beh, row.feat, CONDITION_MAPS)
            res["feat"] = row.feat
            res["session"] = session
            stats_res.append(res)
    stats_res = pd.concat(stats_res)

    # HACK: create a shuffle with just copying the results and labeling them all as shuffle...
    shuffle_vals = stats_res["change in prob"].copy()

    plot_choice_reward(stats_res, subject, shuffle_vals)
    # plot_full_choice_reward(stats_res, subject)
    plot_chose_reward_by_partitions(stats_res, "correct", subject, shuffle_vals)
    plot_chose_reward_by_partitions(stats_res, "incorrect", subject, shuffle_vals)

if __name__ == "__main__":
    main()
