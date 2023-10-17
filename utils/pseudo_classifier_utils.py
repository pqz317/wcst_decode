import numpy as np
from sklearn.model_selection import train_test_split
from trial_splitters.trial_splitter import TrialSplitter
from models.wcst_dataset import WcstDataset
import pandas as pd
import copy


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



def evaluate_classifiers_by_time_bins(model, sess_datas, time_bins, num_splits, num_train_per_cond=2000, num_test_per_cond=400, seed=42):
    training_accs_by_bin = np.empty((len(time_bins), num_splits))
    test_accs_by_bin = np.empty((len(time_bins), num_splits))
    shuffled_accs_by_bin = np.empty((len(time_bins), num_splits))
    models_by_bin = np.empty((len(time_bins), num_splits), dtype=object)
    rng = np.random.default_rng(seed)
    for time_bin_idx, time_bin in enumerate(time_bins): 
        print(f"Working on bin {time_bin}")
        for split_idx in range(num_splits):
            pseudo_sess = pd.concat(sess_datas.apply(
                lambda x: x.generate_pseudo_data(num_train_per_cond, num_test_per_cond, time_bin)
            ).values, ignore_index=True)

            train_data = pseudo_sess[pseudo_sess.Type == "Train"]
            test_data = pseudo_sess[pseudo_sess.Type == "Test"]

            x_train = transform_input_data(train_data)
            y_train = transform_label_data(train_data)

            x_test = transform_input_data(test_data)
            y_test = transform_label_data(test_data)

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
