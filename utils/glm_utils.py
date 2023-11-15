import pandas as pd
import numpy as np

from models.trainer import Trainer
from models.model_wrapper import ModelWrapperRegression
from models.multinomial_logistic_regressor import MultinomialLogisticRegressor
from torch.nn import PoissonNLLLoss
from sklearn.linear_model import PoissonRegressor

def fit_glm_torch(df, x_cols, y_col):
    ys = df[y_col].values
    xs = df[x_cols].values
    init_params = {"n_inputs": xs.shape[1], "n_classes": 1}
    # create a trainer object
    trainer = Trainer(
        learning_rate=0.05, 
        max_iter=500, 
        loss_fn=PoissonNLLLoss(log_input=True),
        weight_decay=1
    )
    # create a wrapper for the decoder
    model = ModelWrapperRegression(MultinomialLogisticRegressor, init_params, trainer)
    model = model.fit(xs, ys)
    return pd.Series({"score": model.score(xs, ys)})

def fit_glm(df, x_cols, y_col):
    ys = df[y_col].values
    if np.all(ys == 0): 
        print("All 0 frs, skipping fitting")
        return pd.Series({"score": 0.0})
    xs = df[x_cols].values
    model = PoissonRegressor(alpha=1)
    model = model.fit(xs, ys)
    return pd.Series({"score": model.score(xs, ys)})

def flatten_columns(beh, columns):
    flattened_columns = []
    for column in columns:
        values = beh[column].unique()
        for value in values:
            beh[value] = (beh[column] == value).astype(int)
        flattened_columns.extend(values)
    return beh, flattened_columns

def create_shuffles(data, columns, rng):
    for column in columns:
        vals = data[column].values
        rng.shuffle(vals)
        data[column] = vals
    return data

def fit_glms_by_unit_and_time(data, x_inputs):
    data, flattened_columns = flatten_columns(data, x_inputs)
    res = data.groupby(["UnitID", "TimeBins"]).apply(lambda x: fit_glm(x, flattened_columns, "SpikeCounts")).reset_index()
    return res.fillna(0)

def fit_glm_for_data(data, input_columns):
    beh, frs = data
    beh_inputs = beh[input_columns]
    data = pd.merge(beh_inputs, frs, on="TrialNumber")
    res = fit_glms_by_unit_and_time(data, input_columns)
    return res

def get_sig_bound(group, p_val, num_hyp):
    percentile = (1 - p_val / num_hyp) * 100
    return pd.Series({"sig_bound": np.percentile(group.score, percentile, method='higher')})

def calculate_sig_stats(shuffled, p_val, num_hyp):
    stats = shuffled.groupby(["UnitID", "TimeBins"]).apply(lambda group: get_sig_bound(group, p_val, num_hyp)).reset_index()
    return stats

def identify_significant_units(res, shuffled_res, time_idxs, alpha=0.05):
    stats = calculate_sig_stats(shuffled_res, alpha, len(time_idxs))
    merged = pd.merge(res, stats, on=["UnitID", "TimeBins"])
    # TODO: HACK!! make general solution here, get rid of TimeBins disaster
    merged["TimeIdxs"] = (merged["TimeBins"] * 10).astype(int)
    merged = merged[merged.TimeIdxs.isin(time_idxs)]
    def assess_unit(unit_group):
        sig = unit_group.score > unit_group.sig_bound
        return pd.Series({"IsSig": np.any(sig)})
    return merged.groupby("UnitID").apply(assess_unit).reset_index()