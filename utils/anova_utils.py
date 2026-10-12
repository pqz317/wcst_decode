import utils.behavioral_utils as behavioral_utils
import numpy as np
import pandas as pd
import itertools
from constants.decoding_constants import SESS_SPIKES_PATH
import utils.io_utils as io_utils

def compute_average_over(df, value_col, groupby_cols, name):
    avg = df.groupby(groupby_cols)[value_col].mean().reset_index(name=name)
    return pd.merge(df, avg, on=groupby_cols)

def anova_factors(df, conditions):
    """
    Iteratively go through combinations of conditions, compute means, subtract them out
    """
    df = compute_average_over(df, "FiringRate", ["PseudoUnitID"], "x")
    df["residual"] = df.FiringRate - df.x
    for comb_num in range(1, len(conditions)+1):
        combs = list(itertools.combinations(conditions, r=comb_num))
        for comb in combs:
            comb_str = "x_" +"".join(comb)
            df = compute_average_over(df, "residual", ["PseudoUnitID"] + list(comb), comb_str)
        df["residual"] = df["residual"] - df[["x_" + "".join(comb) for comb in combs]].sum(axis=1)
    return df

def calc_unit_var(unit_df, conditions):
    """
    For a single unit, compute the fraction of variance
    explained for each of the condition combinations
    Args: 
    - unit_df: unit firing rates mean subtract by conditions
    - conditions: list of conditions to compute variance for
    Returns: 
    - pd series of fraction of variances
    """
    row = {}
    sum = 0.0
    total_var = unit_df.FiringRate.var()
    row["total_var"] = total_var
    combs = get_combs_of_conds(conditions)
    for comb in combs:
        comb_str = "x_" + "".join(comb)
        var_frac = unit_df[comb_str].var() / total_var
        row[f"{comb_str}_fracvar"] = var_frac
        sum += var_frac
    row["residual_fracvar"] = unit_df.residual.var() / total_var
    sum += row["residual_fracvar"]
    row["sum_fracvar"] = sum
    return pd.Series(row)

def cell_fracvar(X, dim_idx, window_bins):
    """
    Fraction of variance explained by a two-level factor plus its interaction with time, per unit and
    window. Equal to x_{factor}_comb_time_fracvar of anova_factors -> calc_unit_var ->
    combine_time_fracvar with conditions [TimeBins, factor], but computed from cell means:
    the factor and factor x time terms together are sum_{d,b} n_d (m_db - m_b)^2, where m_db is the
    mean over level-d trials in bin b, m_b the mean over all trials in bin b, n_d the level-d trial
    count. The cross terms cancel since every trial has every bin.
    Args:
    - X: firing rates, trials x units x bins
    - dim_idx: factor level, 0 or 1, per trial
    - window_bins: list of arrays of bin indices, one per window
    Returns:
    - fracvar, total_var: each units x windows. total_var is the variance over (trial, bin) rows,
      ddof 1, as in calc_unit_var
    """
    n_d = np.array([(dim_idx == d).sum() for d in (0, 1)])
    sums_d = np.stack([X[dim_idx == d].sum(axis=0) for d in (0, 1)])
    sums = sums_d.sum(axis=0)
    sq_sums = (X ** 2).sum(axis=0)
    n = n_d.sum()
    fracvar, total_var = [], []
    for bins in window_bins:
        m_db = sums_d[:, :, bins] / n_d[:, None, None]
        m_b = sums[:, bins] / n
        between = (n_d[:, None, None] * (m_db - m_b) ** 2).sum(axis=(0, 2))
        num_rows = n * len(bins)
        ss_tot = sq_sums[:, bins].sum(axis=1) - sums[:, bins].sum(axis=1) ** 2 / num_rows
        fracvar.append(between / ss_tot)
        total_var.append(ss_tot / (num_rows - 1))
    return np.stack(fracvar, axis=1), np.stack(total_var, axis=1)

def get_combs_of_conds(conditions):
    """
    Helper to return a list of all possible combinations of the conditions list
    """
    combs = []
    for comb_num in range(1, len(conditions)+1):
        combs.extend(itertools.combinations(conditions, r=comb_num))
    return combs

def combine_time_fracvar(unit_vars, conditions):
    """
    Combines time component

    Ex: 
    - for condition C
    - unit_var contains columns x_C_fracvar, x_TimeBins_C_fracvar
    - adds a column for x_C_comb_time_fracvar which is the sum of the two. 
    """
    combs = get_combs_of_conds(conditions)
    for comb in combs: 
        combined_cond_str = "".join(comb)
        unit_vars[f"x_{combined_cond_str}_comb_time_fracvar"] = unit_vars[f"x_{combined_cond_str}_fracvar"] + unit_vars[f"x_TimeBins{combined_cond_str}_fracvar"]
    return unit_vars

# def anova_session(row, feat, conditions, trial_interval):
#     """
#     return df with columns: pseudo unit id, time_bin, response, choice, firing_rate
#     """
#     df = load_data(row.session_name, feat, trial_interval)
#     df = anova_factors(df, conditions)

#     unit_vars = df.groupby("PseudoUnitID").apply(lambda x: cal_unit_var(x, conditions)).reset_index()
#     return unit_vars


def compute_total_frac_vars(x, vars):
    total_pop_var = x.total_var.sum()
    return pd.Series({m: (x[f"x_{m}_comb_time_fracvar"] * x.total_var).sum() / total_pop_var for m in vars})

def frac_var_explained_by_time(args, vars, units=None, trial_events=["StimOnset", "FeedbackOnsetLong"], subjects=["SA", "BL"]):
    """
    Computes the fraction of variance explained by some variable of interest, in a given populatio
    Args: 
    - args: 
    - units: list of PseudoUnitIDs to examine
    - vars: variables to to compute frac variance for
    """
    all_res = []
    for var in vars: 
        for trial_event in trial_events:
            for sub in subjects:
                args.subject = sub
                args.trial_event = trial_event
                res = io_utils.read_anova_good_units(args, "all", var)
                res["trial_event"] = trial_event
                res["subject"] = sub
                res["var"] = var
                all_res.append(res)
    all_res = pd.concat(all_res)
    if units is not None: 
        all_res = all_res[all_res.PseudoUnitID.isin(units)]

    total_fracs = all_res.groupby(["feat", "WindowEndMilli", "trial_event"]).apply(lambda x: compute_total_frac_vars(x, vars)).reset_index()
    total_fracs = pd.melt(total_fracs, id_vars=["feat", "WindowEndMilli", "trial_event"], value_vars=vars, var_name="partition_type", value_name="frac_variance")
    return total_fracs
    
def num_sig_units_by_time(args, vars, sig_thresh="95th", units=None, trial_events=["StimOnset", "FeedbackOnsetLong"], subjects=["SA", "BL"]):
    all_res = []
    for var in vars: 
        for trial_event in trial_events:
            for sub in subjects:
                args.subject = sub
                args.trial_event = trial_event
                res = io_utils.read_anova_good_units(args, "all", var)
                res["trial_event"] = trial_event
                res["subject"] = sub
                res["var"] = var
                all_res.append(res)
    all_res = pd.concat(all_res)
    if units is not None: 
        all_res = all_res[all_res.PseudoUnitID.isin(units)]
    units = all_res.PseudoUnitID.unique()
    sig_res = all_res[all_res.p == sig_thresh]
    num_units = sig_res.groupby(["WindowEndMilli", "feat", "trial_event", "var"]).PseudoUnitID.nunique().reset_index(name="num_units")
    num_units["frac_units_sig"] = num_units["num_units"] / len(units)
    return num_units

    