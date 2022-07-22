import numpy as np
from sklearn.model_selection import train_test_split
from trial_splitters.trial_splitter import TrialSplitter
import pandas as pd

def transform_to_input_data(firing_rates, trials_filter=None):
    """Transform DataFrame with columns TrialNumber, UnitID, TimeBins, Value
    to a num_trials x num_inputs numpy array for classifier input

    Args:
        firing_rates: Dataframe with columns: TrialNumber, UnitID,
            TimeBins, Value
        trials_filter: List of trial numbers, which trials to filter on.

    Returns:
        np array of num_trials x num_inputs
    """
    if trials_filter is not None: 
        firing_rates = firing_rates[firing_rates["TrialNumber"].isin(trials_filter)]
    num_time_bins = len(firing_rates["TimeBins"].unique())
    num_units = len(firing_rates["UnitID"].unique())
    num_trials = len(firing_rates["TrialNumber"].unique())
    num_inputs = num_time_bins * num_units
    sorted_by_trial = firing_rates.sort_values(by=["TrialNumber"])
    return sorted_by_trial["Value"].to_numpy().reshape((num_trials, num_inputs))


def transform_to_label_data(feature_selections, trials_filter=None):
    """Transform DataFrame with columns TrialNumber, Feature into numpy array of features

    Args:
        feature_selections: Dataframe with columns: TrialNumber, Feature
        trials_filter: List of trial numbers, which trials to filter on.

    Returns:
        np array of num_trials x 1
    """
    if trials_filter is not None:
        feature_selections = feature_selections[feature_selections["TrialNumber"].isin(trials_filter)]
    sorted_by_trial = feature_selections.sort_values(by=["TrialNumber"])
    return sorted_by_trial["Feature"].to_numpy()


def evaluate_classifier(clf, firing_rates, feature_selections, trial_splitter, seed=10):
    """Given classifier, inputs, and labels, evaluate it with the trial splitter.

    Args:
        firing_rates: Dataframe with columns: TrialNumber, UnitID,
            TimeBins, Value, used as inputs.
        feature_selections: Dataframe with columns: TrialNumber,
            Feature, used as labels

    Returns:
        Tuple of Lists, including test accuracies, training accuracies,
        shuffled accuracies and models.
    """
    test_accs = []
    train_accs = []
    shuffled_accs = []
    models = []
    rng = np.random.default_rng(seed)
    for train_trials, test_trials in trial_splitter:
        x_train = transform_to_input_data(firing_rates, trials_filter=train_trials)
        y_train = transform_to_label_data(feature_selections, trials_filter=train_trials)

        x_test = transform_to_input_data(firing_rates, trials_filter=test_trials)
        y_test = transform_to_label_data(feature_selections, trials_filter=test_trials)
        clf = clf.fit(x_train, y_train)

        train_acc = clf.score(x_train, y_train)
        test_acc = clf.score(x_test, y_test)

        train_accs.append(train_acc)
        test_accs.append(test_acc)

        y_test_shuffle = y_test
        rng.shuffle(y_test_shuffle)
        shuffled_acc = clf.score(x_test, y_test_shuffle)
        shuffled_accs.append(shuffled_acc)
        
        models.append(clf)
        
    return np.array(train_accs), np.array(test_accs), np.array(shuffled_accs), np.array(models)

def evaluate_classifiers_by_time_bins(clf, inputs, labels, time_bins, splitter):
    test_accs_by_bin = np.empty((len(time_bins), len(splitter)))
    shuffled_accs_by_bin = np.empty((len(time_bins), len(splitter)))
    for i, bin in enumerate(time_bins):
        # need isclose because the floats get stored weird
        inputs_for_bin = inputs[np.isclose(inputs["TimeBins"], bin)]
        train_accs, test_accs, shuffled_accs, models = evaluate_classifier(
            clf, inputs_for_bin, labels, splitter
        )
        test_accs_by_bin[i, :] = test_accs
        shuffled_accs_by_bin[i, :] = shuffled_accs
    return test_accs_by_bin, shuffled_accs_by_bin
