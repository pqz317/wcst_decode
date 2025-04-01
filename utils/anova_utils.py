import utils.behavioral_utils as behavioral_utils
import pandas as pd
import itertools
from constants.decoding_constants import SESS_SPIKES_PATH

def load_data(sess_name, feat, trial_interval, subject="SA", unit_id=None):
    beh = behavioral_utils.get_valid_belief_beh_for_sub_sess(subject, sess_name)
    beh = behavioral_utils.get_chosen_single(feat, beh)
    spikes_path = SESS_SPIKES_PATH.format(
        sub=subject,
        sess_name=sess_name, 
        fr_type="firing_rates",
        pre_interval=trial_interval.pre_interval, 
        event=trial_interval.event, 
        post_interval=trial_interval.post_interval, 
        interval_size=trial_interval.interval_size
    )
    frs = pd.read_pickle(spikes_path)
    frs["PseudoUnitID"] = int(sess_name) * 100 + frs.UnitID.astype(int)
    frs = frs[frs.TimeBins > 1.8]
    if unit_id is not None:
        frs = frs[frs.PseudoUnitID == unit_id]
    df = pd.merge(frs, beh, on="TrialNumber")
    return df

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
    row = {}
    sum = 0.0
    total_var = unit_df.FiringRate.var()
    row["total_var"] = total_var
    for comb_num in range(1, len(conditions)+1):
        combs = itertools.combinations(conditions, r=comb_num)
        for comb in combs:
            comb_str = "x_" +"".join(comb)
            var_frac = unit_df[comb_str].var() / total_var
            row[f"{comb_str}_fracvar"] = var_frac
            sum += var_frac
    row["residual_fracvar"] = unit_df.residual.var() / total_var
    sum += row["residual_fracvar"]
    row["sum_fracvar"] = sum
    return pd.Series(row)

def anova_session(row, feat, conditions, trial_interval):
    """
    return df with columns: pseudo unit id, time_bin, response, choice, firing_rate
    """
    df = load_data(row.session_name, feat, trial_interval)
    df = anova_factors(df, conditions)

    unit_vars = df.groupby("PseudoUnitID").apply(lambda x: cal_unit_var(x, conditions)).reset_index()
    return unit_vars
    
    