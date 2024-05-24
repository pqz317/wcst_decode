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
from constants.behavioral_constants import *

def get_data(df, mode, x_cols, trials):
    df_trials = df.loc[trials]
    ys = df_trials[mode].values
    xs = df_trials[x_cols].values
    return xs, ys

def fit_glm(df, x_cols, mode=MODE, model_type=MODEL, include_predictions=INCLUDE_PREDICTIONS, train_test_split=None):
    if train_test_split is not None:
        train_trials, test_trials = train_test_split
    else: 
        train_trials = df.index
        test_trials = df.index
    train_xs, train_ys = get_data(df, mode, x_cols, train_trials)
    test_xs, test_ys = get_data(df, mode, x_cols, test_trials)

    if model_type == "Ridge":
        model = Ridge(alpha=1)
    elif model_type == "Linear":
        model = LinearRegression()
    elif model_type == "LinearNoInt":
        model = LinearRegression(fit_intercept=False)
    elif model_type == "Poisson":
        model = PoissonRegressor(alpha=1)
    else:
        raise ValueError(f"MODEL is specified as {model_type}, invalid value")
    # if np.all(train_ys == 0): 
    #     print("All 0 frs, skipping fitting")
    #     coefs = {f"{col}_coef": 0 for col in x_cols}
    #     if include_predictions:
    #         predictions = np.zeros(len(ys))
    #         return pd.DataFrame({"TrialNumber": df.index, "score": 0.0, "predicted": predictions, "actual": ys} | coefs)
    #     else: 
    #         return pd.Series({"score": 0.0} | coefs)
    model = model.fit(train_xs, train_ys)
    train_score = model.score(train_xs, train_ys)
    train_predictions = model.predict(train_xs)
    test_score = model.score(test_xs, test_ys)
    test_predictions = model.predict(test_xs)
    # df index is TrialNumber
    coefs = {f"{col}_coef": model.coef_[i] for i, col in enumerate(x_cols)}
    if train_test_split is None: 
        if include_predictions:
            res = pd.DataFrame({
                "TrialNumber": df.index, 
                "score": train_score, 
                "predicted": train_predictions, 
                "actual": train_ys
            } | coefs)
        else: 
            res = pd.Series({"score": train_score} | coefs)
    else:
        if include_predictions:
            train_df = pd.DataFrame({"TrialNumber": train_trials, "train_score": train_score, "test_score": test_score, "predicted": train_predictions, "actual": train_ys})
            test_df = pd.DataFrame({"TrialNumber": test_trials, "train_score": train_score, "test_score": test_score, "predicted": test_predictions, "actual": test_ys})
            res = pd.concat((train_df, test_df))
        else: 
            res = pd.Series({"train_score": train_score, "test_score": test_score} | coefs)
    return res

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

def fit_glms_by_unit_and_time(data, input_columns, mode=MODE, model_type=MODEL, include_predictions=INCLUDE_PREDICTIONS, columns_to_flatten=None, train_test_split=None):
    columns_to_flatten = input_columns if columns_to_flatten is None else columns_to_flatten
    not_flattened_columns = [col for col in input_columns if col not in columns_to_flatten]
    data, flattened_columns = flatten_columns(data, columns_to_flatten)
    glm_columns = flattened_columns + not_flattened_columns
    res = data.groupby(["UnitID", "TimeBins"]).apply(
        lambda x: fit_glm(x, glm_columns, mode, model_type, include_predictions, train_test_split)
    ).reset_index()
    return res.fillna(0)

def fit_glm_for_data(
    data, 
    input_columns, 
    mode=MODE, 
    model_type=MODEL, 
    include_predictions=INCLUDE_PREDICTIONS, 
    columns_to_flatten=None, 
    train_test_split=None
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
    res = fit_glms_by_unit_and_time(data, input_columns, mode, model_type, include_predictions, columns_to_flatten, train_test_split)
    return res

def get_sig_bound(group, alpha, num_hyp, score_type="score"):
    percentile = (1 - alpha / num_hyp) * 100
    return pd.Series({"sig_bound": np.percentile(group[score_type], percentile, method='higher')})

def calculate_sig_stats(shuffled, alpha, num_hyp, score_type="score"):
    stats = shuffled.groupby(["UnitID", "TimeBins"]).apply(lambda group: get_sig_bound(group, alpha, num_hyp, score_type)).reset_index()
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

def calc_normalized_value_coefs(res, value_beh):
    """
    calculated value coeficients that are specific to that are
    normalized to that session's feature value std
    Normalized coef = coef * value's variance
    """
    for feature in FEATURES:
        std = np.std(value_beh[feature + "Value"])
        res[feature + "Value_coef_normed"] = res[feature + "Value_coef"] * std
    return res