import numpy as np
from sklearn.model_selection import train_test_split
from trial_splitters.trial_splitter import TrialSplitter
from models.wcst_dataset import WcstDataset
import pandas as pd
import copy

FEATURES = [
    'CIRCLE', 'SQUARE', 'STAR', 'TRIANGLE', 
    'CYAN', 'GREEN', 'MAGENTA', 'YELLOW', 
    'ESCHER', 'POLKADOT', 'RIPPLE', 'SWIRL'
]

HUMAN_FEATURES = [
    'Q', 'C', 'T', 'S',
    'M', 'Y', 'B', 'G', 
    'R', 'P', 'Z', 'L',
]


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
    sorted_by_trial = firing_rates.sort_values(by=["TrialNumber", "UnitID"])
    return sorted_by_trial["Value"].to_numpy().reshape((num_trials, num_inputs))
    

def transform_cards_or_none(cards_by_trial, trials_filter=None):
    """Transform DataFrame with columns TrialNumber, Item<Index of Card><Feature Dimension>s
    to a num_trials x num_cards x num_features numpy array, with each element representing
    a feature as the 0 - 11 index of a FEATURES constant array. 

    Args:
        card_by_trial: described above
        trials_filter: List of trial numbers, which trials to filter on.

    Returns: 
        np array of num_trials x num_cards (4) x num_features (3)
    """
    if cards_by_trial is None:
        return None

    if trials_filter is not None: 
        cards_by_trial = cards_by_trial[cards_by_trial["TrialNumber"].isin(trials_filter)]
    # create an array of falses
    cards = np.zeros((len(cards_by_trial), 4, 12), dtype=int)
    for card_idx in range(4):
        for dim in ["Color", "Shape", "Pattern"]:
            feature_names = cards_by_trial[f"Item{card_idx}{dim}"]
            features_idx = feature_names.apply(lambda f: FEATURES.index(f))
            # for each trial, make the corresponding feature from features_idx at trial idx 1. 
            cards[np.arange(len(cards_by_trial)), card_idx, features_idx] = 1
    return cards


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




def evaluate_classifier(clf, firing_rates, feature_selections, trial_splitter, cards=None, seed=10):
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
        cards_train = transform_cards_or_none(cards, trials_filter=train_trials)
        y_train = transform_to_label_data(feature_selections, trials_filter=train_trials)

        x_test = transform_to_input_data(firing_rates, trials_filter=test_trials)
        cards_test = transform_cards_or_none(cards, trials_filter=test_trials)
        y_test = transform_to_label_data(feature_selections, trials_filter=test_trials)
        # print("|||||||NEW SPLIT ||||||||||")
        clf = clf.fit(x_train, y_train, cards_train)
        train_acc = clf.score(x_train, y_train, cards_train)
        # print(f"Train Score: {train_acc}")

        # to account for the fact that certain splitters with certain
        # filters may result in no test data. 
        if len(y_test) > 0 and len(x_test) > 0:
            test_acc = clf.score(x_test, y_test, cards_test)
            # print(f"Test Score: {test_acc}")
        else:
            test_acc = np.nan

        train_accs.append(train_acc)
        test_accs.append(test_acc)

        y_test_shuffle = y_test
        rng.shuffle(y_test_shuffle)
        shuffled_acc = clf.score(x_test, y_test_shuffle, cards_test)
        shuffled_accs.append(shuffled_acc)
        # print(f"Shuffled Score: {shuffled_acc}")

        # needed so that every element in models is 
        # from a different instance of the model
        models.append(copy.deepcopy(clf))
        
    return np.array(train_accs), np.array(test_accs), np.array(shuffled_accs), np.array(models)


def evaluate_classifiers_by_time_bins(clf, inputs, labels, time_bins, splitter, cards=None):
    """For every time bin, separately trains/tests classifiers based on the splitter, 
    And returns back a distribution of accuracies for that time bin. 
    Args:
        clf: classifier to perform classification with
        inputs: df with columns: TimeBins, TrialNumber, UnitID, Value
        labels: df with columns: TrialNumber, Feature
        time_bins: bins to evaluate across, units matching TimeBins column in inputs
        splitter: method of splitting data points into training/test

    Returns:
        test_accuracies_by_bin: np array of num_time_bins x num_splits 
        shuffled_accuracies_by_bin: np array of num_time_bins x num_splits 
        trained_models_by_bin: np array of num_time_bins x num_splits of model objects
        splits: list of tuples, each element containing train and test lists of IDs
    """
    training_accs_by_bin = np.empty((len(time_bins), len(splitter)))
    test_accs_by_bin = np.empty((len(time_bins), len(splitter)))
    shuffled_accs_by_bin = np.empty((len(time_bins), len(splitter)))
    models_by_bin = np.empty((len(time_bins), len(splitter)), dtype=object)

    # ensure that every time bin has the same set of train/test splits, 
    splits = [(train, test) for train, test in splitter]
    for i, bin in enumerate(time_bins):
        print(f"Evaluating for bin {bin}")
        # need isclose because the floats get stored weird
        inputs_for_bin = inputs[np.isclose(inputs["TimeBins"], bin)]
        training_accs, test_accs, shuffled_accs, models = evaluate_classifier(
            clf, inputs_for_bin, labels, splits, cards=cards
        )
        # print(training_accs_by_bin.shape)
        # print(training_accs.shape)
        training_accs_by_bin[i, :] = training_accs
        test_accs_by_bin[i, :] = test_accs
        shuffled_accs_by_bin[i, :] = shuffled_accs
        models_by_bin[i, :] = models

    return training_accs_by_bin, test_accs_by_bin, shuffled_accs_by_bin, models_by_bin, splits

def cross_evaluate_by_time_bins(models_by_bin, inputs, labels, splits, input_bins, cards=None):
    """
    For each time bin, evaluate models trained on that time bin against 
    data in other time bins 
    Args:
        models_by_bin: np array of num_model_time_bins x num_splits of model objects
        inputs: df with columns: TimeBins, TrialNumber, UnitID, Value
        labels: df with columns: TrialNumber, Feature
        splits: list of tuples, each element containing train and test lists of IDs
        input_bins: bins to evaluate across, units matching TimeBins column in inputs

    Returns:
        cross_accs: np array of num_model_time_bins x num_input_time_bins, average accuracy of
            models in row time bin evaluated on data from column time bin
    """
    num_model_time_bins = models_by_bin.shape[0]
    cross_accs = np.empty((num_model_time_bins, len(input_bins)))
    for model_bin_idx in range(num_model_time_bins):
        models = models_by_bin[model_bin_idx, :]
        for test_bin_idx, timebin in enumerate(input_bins):
            accs = []
            inputs_for_bin = inputs[np.isclose(inputs["TimeBins"], timebin)]
            for split_idx, model in enumerate(models):
                # assumes models, splits are ordered the same
                if splits:
                    _, tests = splits[split_idx]
                    trials_filter = tests
                else:
                    trials_filter = None
                x_test = transform_to_input_data(inputs_for_bin, trials_filter=trials_filter)
                cards_test = transform_cards_or_none(cards, trials_filter=trials_filter)
                y_test = transform_to_label_data(labels, trials_filter=trials_filter)
                accs.append(model.score(x_test, y_test, cards_test))
            avg_acc = np.mean(accs)
            cross_accs[model_bin_idx, test_bin_idx] = avg_acc
    return cross_accs

def evaluate_models_by_time_bins(models_by_bin, inputs, labels, bins):
    """
    For each time bin, evaluate models trained on that time bin against a specific set of inputs/labels
    Args:
        models_by_bin: np array of num_time_bins x num_splits of model objects
        inputs: df with columns: TrialNumber, UnitID, Value
        labels: df with columns: TrialNumber, Feature
        bins: bins to evaluate across, units matching TimeBins column in inputs

    Returns:
        accs: np array of num_time_bins x num_models, accuracies of models in time bin evaluated on input/labels
    """  
    accs = np.empty((len(bins), models_by_bin.shape[1]))
    for model_bin_idx in range(len(bins)):
        models = models_by_bin[model_bin_idx, :]
        for idx, model in enumerate(models):
            # assumes models, splits are ordered the same
            inputs_for_bin = inputs[np.isclose(inputs["TimeBins"], bins[model_bin_idx])]
            x_test = transform_to_input_data(inputs_for_bin)
            y_test = transform_to_label_data(labels)
            accs[model_bin_idx, idx] = model.score(x_test, y_test)
    return accs

def evaluate_model_by_training_epoch(wrapper, splitter, inputs, labels, cards=None):
    """For a given time bin, evaluate model performance as a function of training epoch
    NOTE: incompatible with sklearn models
    """
    for train_trials, test_trials in splitter:
        model = wrapper.model_type(**wrapper.init_params)

        x_train = transform_to_input_data(inputs, trials_filter=train_trials)
        cards_train = transform_cards_or_none(cards, trials_filter=train_trials)
        y_train = transform_to_label_data(labels, trials_filter=train_trials)
        y_train_idxs = np.array([wrapper.labels_to_idx[label] for label in y_train.tolist()]).astype(int)

        x_test = transform_to_input_data(inputs, trials_filter=test_trials)
        cards_test = transform_cards_or_none(cards, trials_filter=test_trials)
        y_test = transform_to_label_data(labels, trials_filter=test_trials)
        y_test_idxs = np.array([wrapper.labels_to_idx[label] for label in y_test.tolist()]).astype(int)
        dataset = WcstDataset(x_train, y_train_idxs, cards_train)
        losses, intermediates = wrapper.trainer.train(model, dataset)
        for int_model in intermediates:
            int_model()
    pass


def evaluate_model_weights_by_time_bins(models_by_bin, num_neurons, num_classes):
    """For each time bin, look at the weights of models of that bin

    Args: 
        models_by_bin: np array of num_time_bins x num_splits of model objects
    
    Returns:
        weights: np array of num_neurons, num_time_bins
    """

    # create weight matrix of num_time_bins, num_splits, num_neurons, num_classes
    weights_mat = np.empty((models_by_bin.shape[0], models_by_bin.shape[1], num_neurons, num_classes))
    # populate
    for time_idx, splits_idx in np.ndindex(models_by_bin.shape):
        model = models_by_bin[time_idx, splits_idx]
        # weights in num_neurons x num_classes
        weights = model.coef_.T
        weights_mat[time_idx, splits_idx, :, :] = weights

    abs_weights = np.abs(weights_mat)

    # num_time_bins, num_neurons, num_classes
    avg_across_splits = np.mean(abs_weights, axis=1)

    # num_time_bins, num_splits, num_neurons
    max_across_classes = np.amax(avg_across_splits, axis=2)

    return max_across_classes.T

def evaluate_model_weight_diffs(models):
    """
    If a binary logistic regressor, compute the difference between w1, w2 for each 
    Returns a weights matrix of [num_time_bins, num_splits, num_neurons]
    """
    if not models[0, 0].coef_.shape[0] == 2: 
        raise ValueError("models must only have 2 classes")
    weights_mat = np.empty((models.shape[0], models.shape[1], models[0, 0].coef_.shape[1]))
    for time_idx, splits_idx in np.ndindex(models.shape):
        model = models[time_idx, splits_idx]
        # weights in num_neurons x num_classes
        weights_diff = model.coef_[0, :] - model.coef_[1, :]
        weights_mat[time_idx, splits_idx, :] = weights_diff
    return weights_mat

def convert_model_weights_to_df(weights, pre_interval, interval_size):
    """
    Converts weights that's num_neurons x num_time_bins into df
    which will have rows UnitID, TimeBin, Weight
    """
    num_neurons, num_time_bins = weights.shape
    print(num_neurons)
    reshaped = np.reshape(weights, num_neurons * num_time_bins)
    idx = np.arange(num_neurons * num_time_bins)
    unit_ids = idx // num_time_bins
    time_bins = (idx % num_time_bins) * 100 + pre_interval
    return pd.DataFrame({
        "UnitID": unit_ids,
        "TimeBin": time_bins,
        "Weight": reshaped,
    })

def assess_significance_bootstrap(acc, shuffled_acc, alpha=0.05 / 12, n=1000):
    rng = np.random.default_rng()
    num_t, num_accs = acc.shape
    mean_diffs = np.mean(acc, axis=1) - np.mean(shuffled_acc, axis=1)

    shuff_mean_diffs = np.empty((num_t, n))
    combined = np.hstack((acc, shuffled_acc))
    for i in np.arange(n):
        rng.shuffle(combined, axis=1)
        shuff_mean_diffs[:, i] = np.mean(combined[:, num_accs:], axis=1) - np.mean(combined[:, :num_accs], axis=1)
    quantiles = np.quantile(shuff_mean_diffs, axis=1, q=(1 - alpha))
    return mean_diffs > quantiles

def get_significant_time_bins(res, condition, alpha=None):
    shuffle_str = f"{condition}_shuffle"

    num_time = res.Time.nunique()
    cond_res = res[res.condition == condition]
    accs = cond_res.sort_values(by="Time").Accuracy.values.reshape(num_time, -1)

    shuf_res = res[res.condition == shuffle_str]
    shuff_accs = shuf_res.sort_values(by="Time").Accuracy.values.reshape(num_time, -1)

    if alpha is None: 
        alpha = 0.05 / num_time
    sig_bins = assess_significance_bootstrap(accs, shuff_accs, alpha)
    time_bins = res.Time.sort_values().unique()
    sig_times = time_bins[sig_bins]
    return sig_times

def cosine_sim(vec_a, vec_b):
    return vec_a.dot(vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))    

def get_cross_time_cosine_sim_of_weights(weights):
    """
    Gets averaged cross-time cosine similarity of weights for selected features results
    For each feature, Time X, Time Y, computes across pairs of runs (train/test splits)
    Then averages across features. 
    
    weights df has: feat, Time, weights
    where weights are np array
    Returns df with rows as Time X, columns as Time Y, with values averaged cosine sim
    """
    def cosine_sims_per_feat(feat_weights):
        merged = pd.merge(feat_weights, feat_weights, how="cross")
        # don't consider cosine sims for the same run?
        merged = merged[merged.run_x != merged.run_y]
        merged["cosine_sim"] = merged.apply(lambda x: cosine_sim(x.weights_x, x.weights_y), axis=1)    
        return merged

    cosine_sims = weights.groupby("feat").apply(cosine_sims_per_feat).reset_index()
    mean_cosines = cosine_sims.groupby(["Time_x", "Time_y"]).cosine_sim.mean().reset_index(name="cosine_sim")
    pivoted = mean_cosines.pivot(index="Time_x", columns="Time_y", values="cosine_sim")
    return pivoted

def get_cross_cond_cosine_sim_of_weights(weights_a, weights_b, merge_on=["Time", "feat"], exclude_same_run=False):
    """
    weights df has: feat, Time, weights
    where weights are np array
    """
    merged = pd.merge(weights_a, weights_b, on=merge_on)
    if exclude_same_run:
        merged = merged[merged.run_x != merged.run_y]
    merged["cosine_sim"] = merged.apply(lambda x: cosine_sim(x.weights_x, x.weights_y), axis=1)
    return merged

def get_weights_dff(models):
    """
    For a binary classifier, 
    """

