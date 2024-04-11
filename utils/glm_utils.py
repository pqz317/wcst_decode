import pandas as pd
import numpy as np

from models.trainer import Trainer
from models.model_wrapper import ModelWrapperRegression
from models.multinomial_logistic_regressor import MultinomialLogisticRegressor
from torch.nn import PoissonNLLLoss
from sklearn.linear_model import (
    PoissonRegressor,
    LinearRegression,
    Ridge,
)
from constants.glm_constants import *

def fit_glm(df, x_cols, mode=MODE, model_type=MODEL, include_predictions=INCLUDE_PREDICTIONS):
    ys = df[mode].values
    if np.all(ys == 0): 
        print("All 0 frs, skipping fitting")
        coefs = {f"{col}_coef": 0 for col in x_cols}
        if include_predictions:
            predictions = np.zeros(len(ys))
            return pd.DataFrame({"TrialNumber": df.index, "score": 0.0, "predicted": predictions, "actual": ys} | coefs)
        else: 
            return pd.Series({"score": 0.0} | coefs)
    xs = df[x_cols].values
    if model_type == "Ridge":
        model = Ridge(alpha=1)
    elif model_type == "Linear":
        model = LinearRegression()
    elif model_type == "Poisson":
        model = PoissonRegressor(alpha=1)
    else:
        raise ValueError(f"MODEL is specified as {model_type}, invalid value")
    model = model.fit(xs, ys)
    score = model.score(xs, ys)
    predictions = model.predict(xs)
    # df index is TrialNumber
    coefs = {f"{col}_coef": model.coef_[i] for i, col in enumerate(x_cols)}
    if include_predictions:
        res = pd.DataFrame({"TrialNumber": df.index, "score": score, "predicted": predictions, "actual": ys} | coefs)
    else: 
        res = pd.Series({"score": score} | coefs)
    return res
    # return pd.DataFrame({"score": model.score(xs, ys), "prediction": model.predict(xs)})

def flatten_columns(beh, columns):
    flattened_columns = []
    for column in columns:
        values = beh[column].unique()
        for value in values:
            beh[value] = (beh[column] == value).astype(int)
        flattened_columns.extend(values)
    return beh, flattened_columns

def create_shuffles(data, columns, rng):
    """
    Shuffles columns specified
    NOTE: ensures that the columns are shuffled together
    """
    shuffled_idxs = np.arange(len(data))
    rng.shuffle(shuffled_idxs)
    for column in columns:
        vals = data[column].values[shuffled_idxs]
        data[column] = vals
    return data

def fit_glms_by_unit_and_time(data, input_columns, mode=MODE, model_type=MODEL, include_predictions=INCLUDE_PREDICTIONS, columns_to_flatten=None):
    columns_to_flatten = input_columns if columns_to_flatten is None else columns_to_flatten
    not_flattened_columns = [col for col in input_columns if col not in columns_to_flatten]
    data, flattened_columns = flatten_columns(data, columns_to_flatten)
    glm_columns = flattened_columns + not_flattened_columns
    res = data.groupby(["UnitID", "TimeBins"]).apply(
        lambda x: fit_glm(x, glm_columns, mode, model_type, include_predictions)
    ).reset_index()
    return res.fillna(0)

def fit_glm_for_data(
    data, input_columns, mode=MODE, model_type=MODEL, include_predictions=INCLUDE_PREDICTIONS, columns_to_flatten=None, 
):
    """
    Fits GLMs for each unit, for each timebin, across trials
    Args: 
        data: tuple of behavior df, firing rate df
        input_columns: list of columns names defining GLM params
        mode: str of whether GLM should be fit on spikes of firing rates
        model_type: str of type of GLM, ex Ridge, Linear, or Poisson
        include_predictions: bool of whether or not to include predicted firing rate for each trial
        columns_to_flatten: list of columns to flatten or one-hot encode, where each unique value within the 
            becomes its own column. 
    """
    
    beh, frs = data
    beh_inputs = beh[input_columns]
    data = pd.merge(beh_inputs, frs, on="TrialNumber")
    res = fit_glms_by_unit_and_time(data, input_columns, mode, model_type, include_predictions, columns_to_flatten)
    return res

def get_sig_bound(group, p_val, num_hyp):
    percentile = (1 - p_val / num_hyp) * 100
    return pd.Series({"sig_bound": np.percentile(group.score, percentile, method='higher')})

def calculate_sig_stats(shuffled, p_val, num_hyp):
    stats = shuffled.groupby(["UnitID", "TimeBins"]).apply(lambda group: get_sig_bound(group, p_val, num_hyp)).reset_index()
    return stats

def identify_significant_units(res, shuffled_res, time_idxs, alpha=0.01):
    stats = calculate_sig_stats(shuffled_res, alpha, len(time_idxs))
    merged = pd.merge(res, stats, on=["UnitID", "TimeBins"])
    # TODO: HACK!! make general solution here, get rid of TimeBins disaster
    merged["TimeIdxs"] = (merged["TimeBins"] * 10).astype(int)
    merged = merged[merged.TimeIdxs.isin(time_idxs)]
    def assess_unit(unit_group):
        sig = unit_group.score > unit_group.sig_bound
        return pd.Series({"IsSig": np.any(sig)})
    return merged.groupby("UnitID").apply(assess_unit).reset_index()