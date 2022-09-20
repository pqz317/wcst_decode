import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import scipy.stats
from lfp_tools import (
    general as lfp_general,
    startup as lfp_startup,
    development as lfp_development,
    analysis as lfp_analysis
)
from spike_tools import (
    general as spike_general,
    analysis as spike_analysis,
)
import s3fs
import utils.behavioral_utils as behavioral_utils
import utils.spike_utils as spike_utils
import utils.classifier_utils as classifier_utils
import utils.visualization_utils as visualization_utils
import utils.io_utils as io_utils
from trial_splitters.random_splitter import RandomSplitter
from trial_splitters.block_splitter import BlockSplitter
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import pickle

from models.value_models import ValueLinearModel, ValueNormedModel
from models.multinomial_logistic_regressor import MultinomialLogisticRegressor, NormedMultinomialLogisticRegressor
from models.model_wrapper import ModelWrapper

from models.trainer import Trainer

import torch

import plotly.express as px


matplotlib.rcParams['figure.dpi'] = 150


species = 'nhp'
subject = 'SA'
exp = 'WCST'
session = 20180802  # this is the session for which there are spikes at the moment. 

feature_dims = ["Color", "Shape", "Pattern"]

pre_interval = 1300
post_interval = 1500


def main():

    # grab behavioral data, spike data, trial numbers. 
    fs = s3fs.S3FileSystem()
    behavior_file = spike_general.get_behavior_path(subject, session)
    behavior_data = pd.read_csv(fs.open(behavior_file))
    valid_beh = behavior_data[behavior_data.Response.isin(["Correct", "Incorrect"])]   

    print("Fetching LFP data")
    lfp = pd.read_csv(fs.open("l2l.jbferre.scratch/for_Patrick/fix.csv"))
    feature_selections = pd.read_pickle(fs.open("l2l.pqz317.scratch/feature_selections.pickle"))
    valid_lfp = lfp[lfp.TrialNumber.isin(valid_beh.TrialNumber.unique())]

    freqs = ["bp_4-5", "bp_10-13", "bp_20-24", "bp_27-37", "bp_65-87", "bp_120-148"]
    feature_dims = ["Shape", "Pattern"]
    for feature_dim in feature_dims:
        for freq in freqs:
            print(f"Running for freq {freq} dimension {feature_dim}")
            pre_interval = 1300
            post_interval = 1500

            num_channels = len(valid_lfp.ChanNum.unique())
            labels = feature_selections[feature_dim].unique()
            init_params = {"n_inputs": num_channels, "n_classes": len(labels)}
            trainer = Trainer(learning_rate=0.05, max_iter=1000)
            wrapped = ModelWrapper(NormedMultinomialLogisticRegressor, init_params, trainer, labels)

            inputs = valid_lfp.rename(columns={freq: "Value", "ChanNum": "UnitID"})
            labels = feature_selections.rename(columns={feature_dim: "Feature"})

            random_splitter = RandomSplitter(labels.TrialNumber.unique(), 20, 0.2)
            train_accs_by_bin, test_accs_by_bin, shuffled_accs, models, splits = classifier_utils.evaluate_classifiers_by_time_bins(
                wrapped, inputs, labels, np.arange(0, 2.8, 0.1), random_splitter
            )
            np.save(fs.open(f"l2l.pqz317.scratch/{feature_dim}_normed_lr_lfp_{freq}_accs_{pre_interval}_fb_{post_interval}_by_bin_random_split.npy", "wb"), test_accs_by_bin)
            np.save(fs.open(f"l2l.pqz317.scratch/{feature_dim}_normed_lr_lfp_{freq}shuffled_accs_{pre_interval}_fb_{post_interval}_by_bin_random_split.npy", "wb"), shuffled_accs)
            np.save(fs.open(f"l2l.pqz317.scratch/{feature_dim}_normed_lr_lfp_{freq}_models_{pre_interval}_fb_{post_interval}_by_bin_random_split.npy", "wb"), models)
            pickle.dump(splits, fs.open(f"l2l.pqz317.scratch/{feature_dim}_normed_lr_lfp_{freq}_splits_{pre_interval}_fb_{post_interval}_by_bin_random_split.npy", "wb"))      




if __name__ == "__main__":
    main()