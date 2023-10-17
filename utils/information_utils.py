import numpy as np
import pandas as pd


def calc_entropy(data):
    """
    Calculates entropy of a given dataset
    """
    _, unique_counts = np.unique(data, return_counts=True)
    probs = unique_counts / len(data)
    entropy = -1 * np.sum(probs * np.log(probs))
    return entropy

def calc_entropy_across_unit_and_time(df, x_column):
    return df.groupby(["UnitID", "TimeBins"]).apply(lambda group: calc_entropy(group[x_column])).to_frame("Entropy")

def calc_conditional_entropy(df, x_column, y_column):
    num_points = len(df)
    per_condition = df.groupby(y_column).apply(
        lambda group: len(group) / num_points * calc_entropy(group[x_column])
    ).to_frame("EntropyPerCond") 
    return per_condition["EntropyPerCond"].sum()


def calc_mutual_information_for_columns(df, x_column, y_columns, h_x=None):
    """
    Calculates mutual information with the relationship:
    I(X; Y) = H(X) - H(X|Y)
    """
    if h_x is None: 
        h_x = calc_entropy(df[x_column])
    mis = []
    for y_column in y_columns:
        h_x_given_y = calc_conditional_entropy(df, x_column, y_column)    
        mis.append(h_x - h_x_given_y)
    return pd.Series({f"MI{y_columns[i]}": mi for i, mi in enumerate(mis)})

def calc_mutual_information_per_unit_and_time(df, x_column, y_columns):
    """
    Calculates mutual information with the relationship:
    I(X; Y) = H(X) - H(X|Y)
    """
    return df.groupby(["UnitID", "TimeBins"]).apply(lambda group: calc_mutual_information_for_columns(group, x_column, y_columns)).reset_index()

def calculate_shuffled_stats(group, columns):
    row = {}
    for column in columns:
        vals = group[f"MIShuffled{column}"]
        row[f"MIShuffled{column}95th"] = np.percentile(vals, 95)
        row[f"MIShuffled{column}99th"] = np.percentile(vals, 99)
        row[f"MIShuffled{column}Mean"] = np.mean(vals)
        row[f"MIShuffled{column}Std"] = np.std(vals)
    return pd.Series(row)

def calculate_bonferroni_corrected_stats(group, columns, p_val, num_hyp):
    row = {}
    for column in columns:
        vals = group[f"MIShuffled{column}"]
        percentile = (1 - p_val / num_hyp) * 100
        row[f"MIShuffled{column}Corrected"] = np.percentile(vals, percentile, method='higher')
    return pd.Series(row)

def assess_unit_significance(unit_mi_df, y_columns):
    """
    For a unit's MI metrics, look at every whether the unit contains signficant information
    on any of the y_column's variables for any time bin. 
    Per y_column, add an additional column for whether unit is significant or not
    Uses Bonferroni Correction for p < 0.05: https://en.wikipedia.org/wiki/Bonferroni_correction 
    Number of hypotheses for correction is determined when metric is calculated
    """
    row = {}
    for y_column in y_columns:
        stat_column = f"MIShuffled{y_column}Corrected"
        sig = unit_mi_df[f"MI{y_column}"] > unit_mi_df[stat_column]
        row[f"{y_column}Sig"] = np.any(sig)
    return pd.Series(row)

def assess_significance(mi_df, y_columns):
    unit_sig = mi_df.groupby("UnitID").apply(lambda group: assess_unit_significance(group, y_columns)).reset_index()
    return unit_sig

def calc_corrected_null_stats(shuffled_mis, y_columns, p_val, num_hyp):
    shuffled_mis.set_index(["UnitID", "TimeBins"])
    null_stats = shuffled_mis.groupby(["UnitID", "TimeBins"]).apply(
        lambda group: calculate_bonferroni_corrected_stats(group, y_columns, p_val, num_hyp)
    ).reset_index()
    return null_stats