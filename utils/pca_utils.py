import scipy
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd


"""
Utils for computing PCA trajectories in pseudo-populations, grouped by some condition
"""

def mean_sub_unit(group):
    mean = group.ConditionedFiringRate.mean()
    group["ConditionedFiringRate"] = group.ConditionedFiringRate - mean
    return group

def make_data_mat(all_trials, condition, should_mean_sub=True):
    """
    Returns 
    - data_mat: [num_units, (num_conds x num_time_bins)]
    - conditioned_frs_sorted: df with condition, PseudoUnitID, TimeBins, ConditionedFiringRate
    """
    conditions = all_trials[condition].unique()
    conditions.sort()
    conditioned_frs = all_trials.groupby([condition, "PseudoUnitID", "TimeBins"]).FiringRate.mean().to_frame("ConditionedFiringRate").reset_index()
    if should_mean_sub: 
        conditioned_frs = conditioned_frs.groupby("PseudoUnitID", group_keys=False).apply(mean_sub_unit).reset_index()
    conditioned_frs_sorted = conditioned_frs.sort_values(by=["PseudoUnitID", condition, "TimeBins"])
    num_units = len(conditioned_frs.PseudoUnitID.unique())
    num_conds = len(conditioned_frs[condition].unique())
    num_time_bins = len(conditioned_frs.TimeBins.unique())
    data_mat = conditioned_frs_sorted.ConditionedFiringRate.values.reshape((num_units, num_conds * num_time_bins))
    return data_mat, conditioned_frs_sorted

def project_conditioned_firing_rates(all_trials, condition):
    """
    Condition-average firing rates by Condition column,
    Calculate 1st 3 PCs of condition-averaged firing rates
    Parameters: 
    - all_trials: df with PseudoUnitID, TimeBins, TrialNumber, condition column, FiringRate
    - condition: name of column to group by
    Returns: projected dataframe, PCA object
    """
    data_mat, conditioned_frs_sorted = make_data_mat(all_trials, condition)

    pca = PCA()
    pca = pca.fit(data_mat.T)
    components = pca.components_

    def transform_pca(group):
        group = group.sort_values(by="PseudoUnitID")
        vec = group.ConditionedFiringRate.values.reshape(-1, 1)
        transformed = components @ vec
        return pd.Series({"PC1": transformed[0, 0], "PC2": transformed[1, 0], "PC3": transformed[2, 0]})
    transformed_df = conditioned_frs_sorted.groupby(["TimeBins", condition]).apply(transform_pca).reset_index()

    return transformed_df, pca