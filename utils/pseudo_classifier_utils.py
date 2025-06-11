import numpy as np
from sklearn.model_selection import train_test_split
from trial_splitters.trial_splitter import TrialSplitter
from models.wcst_dataset import WcstDataset
import pandas as pd
import copy
from tqdm import tqdm

"""
Need to adapt classifier utils due to the pseudo population workflow being so different
"""
def transform_input_data(pseudo_data):
    pseudo_data = pseudo_data[["PseudoTrialNumber", "PseudoUnitID", "Value"]]
    num_units = len(pseudo_data["PseudoUnitID"].unique())
    num_trials = len(pseudo_data["PseudoTrialNumber"].unique())
    sorted_by_trial = pseudo_data.sort_values(by=["PseudoTrialNumber", "PseudoUnitID"])
    return sorted_by_trial["Value"].to_numpy().reshape((num_trials, num_units))

def transform_label_data(pseudo_data):
    pseudo_data = pseudo_data[["PseudoTrialNumber", "Condition"]].drop_duplicates()
    sorted = pseudo_data.sort_values(by=["PseudoTrialNumber"])
    return sorted["Condition"].to_numpy()



def evaluate_classifiers_by_time_bins(
    model, sess_datas, time_bins, num_splits, 
    num_train_per_cond=2000, 
    num_test_per_cond=400, 
    seed=42, 
    proj_matrix=None,
):
    training_accs_by_bin = np.empty((len(time_bins), num_splits))
    test_accs_by_bin = np.empty((len(time_bins), num_splits))
    shuffled_accs_by_bin = np.empty((len(time_bins), num_splits))
    models_by_bin = np.empty((len(time_bins), num_splits), dtype=object)
    rng = np.random.default_rng(seed)
    for time_bin_idx, time_bin in tqdm(enumerate(time_bins)): 
        print(f"Working on bin {time_bin}")
        for split_idx in range(num_splits):
            pseudo_sess = pd.concat(sess_datas.apply(
                lambda x: x.generate_pseudo_data(num_train_per_cond, num_test_per_cond, time_bin, split_idx)
            ).values, ignore_index=True)

            train_data = pseudo_sess[pseudo_sess.Type == "Train"]
            test_data = pseudo_sess[pseudo_sess.Type == "Test"]

            x_train = transform_input_data(train_data)
            y_train = transform_label_data(train_data)

            x_test = transform_input_data(test_data)
            y_test = transform_label_data(test_data)

            if proj_matrix is not None:
                # means = np.mean(x_train, axis=1)
                # stds = np.std(x_train, axis=1)
                # norm_x_train = (x_train - means) / stds
                # norm_x_test = (x_test - means) / stds
                # x_train = norm_x_train @ proj_matrix
                # x_test = norm_x_test @ proj_matrix
                # TODO: chat with about whether we need to normalize first
                x_train = x_train @ proj_matrix
                x_test = x_test @ proj_matrix                

            print("Fitting model")
            model = model.fit(x_train, y_train)
            train_acc = model.score(x_train, y_train)
            test_acc = model.score(x_test, y_test)

            y_test_shuffle = y_test.copy()
            rng.shuffle(y_test_shuffle)
            shuffled_acc = model.score(x_test, y_test_shuffle)

            training_accs_by_bin[time_bin_idx, split_idx] = train_acc
            test_accs_by_bin[time_bin_idx, split_idx] = test_acc
            shuffled_accs_by_bin[time_bin_idx, split_idx] = shuffled_acc
            models_by_bin[time_bin_idx, split_idx] = copy.deepcopy(model)
    return training_accs_by_bin, test_accs_by_bin, shuffled_accs_by_bin, models_by_bin

def cross_evaluate_by_time_bins(models_by_bin, sess_datas, input_bins, num_train_per_cond=0, num_test_per_cond=200, avg=True):
    """
    Cross evaluates different pseudo models on different time bins
    Assumes sess_datas was generated with the same seed, has splits pre_generated
    """
    num_model_time_bins = models_by_bin.shape[0]
    if avg: 
        cross_accs = np.empty((num_model_time_bins, len(input_bins)))
    else: 
        cross_accs = np.empty((num_model_time_bins, len(input_bins), models_by_bin.shape[1]))
    for model_bin_idx in tqdm(range(num_model_time_bins)):
        print(f"evaluating models for bin idx {model_bin_idx}")
        models = models_by_bin[model_bin_idx, :]
        for test_bin_idx, time_bin in enumerate(input_bins):
            print(f"data at time bin: {time_bin}")
            accs = []
            for split_idx, model in enumerate(models):
                # assumes models, splits are ordered the same
                pseudo_sess = pd.concat(sess_datas.apply(
                    lambda x: x.generate_pseudo_data(num_train_per_cond, num_test_per_cond, time_bin, split_idx)
                ).values, ignore_index=True)

                test_data = pseudo_sess[pseudo_sess.Type == "Test"]

                x_test = transform_input_data(test_data)
                y_test = transform_label_data(test_data)
                accs.append(model.score(x_test, y_test))
            if avg: 
                avg_acc = np.mean(accs)
                cross_accs[model_bin_idx, test_bin_idx] = avg_acc
            else:
                cross_accs[model_bin_idx, test_bin_idx, :] = accs
    return cross_accs

def evaluate_model_with_data(models_by_bin, sess_datas, time_bins, num_train_per_cond=0, num_test_per_cond=200):
    """
    Evaluates model with session datas passed in, ideally from a different condition as what the model was trained on
    """
    accs_across_time = np.empty((len(time_bins), models_by_bin.shape[1]))
    for time_bin_idx, time_bin in enumerate(time_bins):
        print(f"evaluating models for bin idx {time_bin_idx}")
        models = models_by_bin[time_bin_idx, :]
        for split_idx, model in enumerate(models):
            # assumes models, splits are ordered the same
            pseudo_sess = pd.concat(sess_datas.apply(
                lambda x: x.generate_pseudo_data(num_train_per_cond, num_test_per_cond, time_bin)
            ).values, ignore_index=True)

            test_data = pseudo_sess[pseudo_sess.Type == "Test"]

            x_test = transform_input_data(test_data)
            y_test = transform_label_data(test_data)
            # print(x_test.shape)
            # print(x_test[:10, :10])
            # print(y_test.shape)
            # print(y_test[:10])
            acc = model.score(x_test, y_test)
            # print("acc: ")
            # print(acc)
            accs_across_time[time_bin_idx, split_idx] = acc
    return accs_across_time