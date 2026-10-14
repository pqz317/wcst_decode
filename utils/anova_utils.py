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

def factorial_fracvar(X, labels, conditions, window_bins):
    """
    The anova_factors -> calc_unit_var decomposition with conditions ["TimeBins"] + conditions, per
    unit and window, computed from cell sums rather than per row. Reproduces its sequential mean
    subtraction exactly: a comb's term is the mean of the residual left by all lower-order combs.
    Every term is constant within a finest cell (condition levels x bin), so all of it follows from
    each finest cell's trial count and per-unit sum, plus each unit's sum of squares for the total
    and residual variances.
    Args:
    - X: firing rates, trials x units x bins
    - labels: dataframe with one column per condition, one row per trial of X
    - conditions: condition columns of labels, not including TimeBins
    - window_bins: list of arrays of bin indices, one per window
    Returns:
    - dict of column name to units x windows array, the columns and order of calc_unit_var:
      total_var, x_{comb}_fracvar per comb, residual_fracvar, sum_fracvar
    """
    if labels[conditions].isna().any().any():
        raise ValueError("NaN condition labels")
    codes = np.stack([pd.factorize(labels[c])[0] for c in conditions], axis=1)
    cell_codes, g = np.unique(codes, axis=0, return_inverse=True)
    g = g.ravel()
    n = np.bincount(g).astype(float)
    num_trials, num_units, num_bins = X.shape
    one_hot = np.zeros((len(n), num_trials))
    one_hot[g, np.arange(num_trials)] = 1
    S = (one_hot @ X.reshape(num_trials, -1)).reshape(len(n), num_units, num_bins)
    SS = (X ** 2).sum(axis=0)

    # per comb: each finest cell's group, and whether bins stay separate
    all_conds = ["TimeBins"] + conditions
    combs = get_combs_of_conds(all_conds)
    comb_groups = []
    for comb in combs:
        idx = [conditions.index(c) for c in comb if c != "TimeBins"]
        _, h = np.unique(cell_codes[:, idx], axis=0, return_inverse=True)
        h = h.ravel()
        M = np.zeros((h.max() + 1, len(n)))
        M[h, np.arange(len(n))] = 1
        comb_groups.append((len(comb), h, M, "TimeBins" in comb))

    res = {"total_var": [], **{f"x_{''.join(comb)}_fracvar": [] for comb in combs}, "residual_fracvar": [], "sum_fracvar": []}
    nw = n[:, None, None]
    # a unit silent through a window has no variance, so its fractions are 0/0 = NaN, as in calc_unit_var
    with np.errstate(divide="ignore", invalid="ignore"):
        for bins in window_bins:
            Sw = S[:, :, bins]
            N = n.sum() * len(bins)
            m = Sw / nw
            SS_w = SS[:, bins].sum(axis=1)
            total_var = (SS_w - Sw.sum(axis=(0, 2)) ** 2 / N) / (N - 1)
            r = m - (Sw.sum(axis=(0, 2)) / N)[None, :, None]
            level_terms = []
            for i, (level, h, M, keep_bins) in enumerate(comb_groups):
                num = np.tensordot(M, nw * r, axes=(1, 0))
                den = M @ n
                if keep_bins:
                    mean = num / den[:, None, None]
                else:
                    mean = (num.sum(axis=2) / (den * len(bins))[:, None])[:, :, None]
                x = np.broadcast_to(mean[h], r.shape)
                var = ((nw * x ** 2).sum(axis=(0, 2)) - (nw * x).sum(axis=(0, 2)) ** 2 / N) / (N - 1)
                res[f"x_{''.join(combs[i])}_fracvar"].append(var / total_var)
                level_terms.append(x)
                # all combs of a level are means of the same residual, subtracted together
                if i + 1 == len(comb_groups) or comb_groups[i + 1][0] != level:
                    r = r - sum(level_terms)
                    level_terms = []
            res_ss = SS_w - (nw * m ** 2).sum(axis=(0, 2)) + (nw * r ** 2).sum(axis=(0, 2))
            res_var = (res_ss - (nw * r).sum(axis=(0, 2)) ** 2 / N) / (N - 1)
            res["total_var"].append(total_var)
            res["residual_fracvar"].append(res_var / total_var)
            res["sum_fracvar"].append(sum(res[f"x_{''.join(comb)}_fracvar"][-1] for comb in combs) + res["residual_fracvar"][-1])
    return {k: np.stack(v, axis=1) for k, v in res.items()}

def shuffle_stats(shuffles, conditions, keys):
    """
    95th and 99th percentiles over shuffles of each x_{comb}_comb_time_fracvar, per keys group, the
    columns of 20250329_aggregate_shuffles.compute_stats. NaNs are skipped, as Series.quantile does.
    Args:
    - shuffles: run_anova results of all shuffles, concatenated
    - conditions: conditions of the run, not including TimeBins
    - keys: e.g. ["PseudoUnitID", "WindowStartMilli"]
    """
    names = ["".join(comb) for comb in get_combs_of_conds(conditions)]
    # HACK: mirrors run_anova.run_anova -- BeliefPref only exists when Conf is a factor too
    if "BeliefPartition" in conditions and "BeliefConf" in conditions:
        names.append("BeliefPref")
    q = shuffles.groupby(keys)[[f"x_{name}_comb_time_fracvar" for name in names]].quantile([0.95, 0.99]).unstack()
    stats = pd.DataFrame(index=q.index)
    for name in names:
        stats[f"{name}_95th"] = q[(f"x_{name}_comb_time_fracvar", 0.95)]
        stats[f"{name}_99th"] = q[(f"x_{name}_comb_time_fracvar", 0.99)]
    return stats.reset_index()

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

    