import numpy as np
from sklearn.model_selection import train_test_split
from trial_splitters.trial_splitter import TrialSplitter
import pandas as pd

def transform_to_input_data(firing_rates, trials_filter=None):
    """
    Transform DataFrame with columns TrialNumber, UnitID, TimeBins, Value
    to a num_trials x num_inputs numpy array for classifier input

    :param firing_rates: Dataframe with columns: TrialNumber, UnitID, TimeBins, Value
    :param trials_filter: List of trial numbers, which trials to filter on. 
    :returns: np array of num_trials x num_inputs
    """
    df = firing_rates
    if trials_filter is not None: 
        df = df[df["TrialNumber"].isin(trials_filter)]
    num_time_bins = len(df["TimeBins"].unique())
    num_units = len(df["UnitID"].unique())
    num_trials = len(df["TrialNumber"].unique())
    num_inputs = num_time_bins * num_units
    sorted = df.sort_values(by=["TrialNumber"])
    inputs = sorted["Value"].to_numpy().reshape((num_trials, num_inputs))
    return inputs


def transform_to_label_data(feature_selections, trials_filter=None):
    """
    Transform DataFrame with columns TrialNumber, Feature into numpy array of features

    :param feature_selections: Dataframe with columns: TrialNumber, Feature
    :param trials_filter: List of trial numbers, which trials to filter on. 
    :returns: np array of num_trials x 1
    """
    df = feature_selections
    if trials_filter is not None:
        df = df[df["TrialNumber"].isin(trials_filter)]
    sorted = df.sort_values(by=["TrialNumber"])
    inputs = sorted["Feature"].to_numpy()
    return inputs


def evaluate_classifier(clf, firing_rates, feature_selections, trial_splitter):
    """
    Given classifier, inputs, and labels, evaluate it with the trial splitter. 

    :param firing_rates: Dataframe with columns: TrialNumber, UnitID, TimeBins, Value, used as inputs. 
    :param feature_selections: Dataframe with columns: TrialNumber, Feature, used as labels
    :returns: Tuple of Lists, including test accuracies, training accuracies, shuffled accuracies and models. 
    """
    test_accs = []
    train_accs = []
    shuffled_accs = []
    models = []
    for train_trials, test_trials in trial_splitter:
        X_train = transform_to_input_data(firing_rates, trials_filter=train_trials)
        y_train = transform_to_label_data(feature_selections, trials_filter=train_trials)

        X_test = transform_to_input_data(firing_rates, trials_filter=test_trials)
        y_test = transform_to_label_data(feature_selections, trials_filter=test_trials)
        clf = clf.fit(X_train, y_train)

        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        train_accs.append(train_acc)
        test_accs.append(test_acc)

        y_test_shuffle = y_test
        np.random.shuffle(y_test_shuffle)
        shuffled_acc = clf.score(X_test, y_test_shuffle)
        shuffled_accs.append(shuffled_acc)
        
        models.append(clf)
        
    return train_accs, test_accs, shuffled_accs, models