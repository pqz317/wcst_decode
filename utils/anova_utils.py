import utils.behavioral_utils as behavioral_utils
import pandas as pd
import itertools
from constants.decoding_constants import SESS_SPIKES_PATH

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

def anova_session(row, feat, conditions, trial_interval):
    """
    return df with columns: pseudo unit id, time_bin, response, choice, firing_rate
    """
    df = load_data(row.session_name, feat, trial_interval)
    df = anova_factors(df, conditions)

    unit_vars = df.groupby("PseudoUnitID").apply(lambda x: cal_unit_var(x, conditions)).reset_index()
    return unit_vars
    
    