# creates spike by trials, firing rates data for a specific interval

import numpy as np
import pandas as pd
from spike_tools import (
    general as spike_general,
    analysis as spike_analysis,
)
import utils.behavioral_utils as behavioral_utils
import utils.spike_utils as spike_utils
import utils.classifier_utils as classifier_utils
import utils.visualization_utils as visualization_utils
from trial_splitters.random_splitter import RandomSplitter
from models.model_wrapper import ModelWrapper
from models.value_models import ValueNormedDropoutModel
from models.trainer import Trainer
import utils.io_utils as io_utils
import matplotlib.pyplot as plt
import matplotlib

import os

SPECIES = 'nhp'
SUBJECT = 'SA'

PRE_INTERVAL = 1300
POST_INTERVAL = 1500
INTERVAL_SIZE = 100
EVENT = "FeedbackOnset"

REPLACE = True

FEATURE_DIMS = ["Color", "Shape", "Pattern"]


def run_decoder(frs, card_idxs, base_dir):
    """
    Configures decoder to be run for each time bin, trains and tests decoders
    Stores results in filesystem. 
    """
    num_neurons = len(frs.UnitID.unique())
    classes = np.sort(card_idxs.ItemChosen.unique())
    init_params = {"n_inputs": num_neurons, "p_dropout": 0.5, "n_values": 12}
    trainer = Trainer(learning_rate=0.05, max_iter=500, batch_size=10000)
    wrapped = ModelWrapper(ValueNormedDropoutModel, init_params, trainer, classes)

    mode = "SpikeCounts"

    # prep data for classification
    inputs = frs.rename(columns={mode: "Value"})
    labels = card_idxs.rename(columns={"ItemChosen": "Feature"})

    random_splitter = RandomSplitter(labels.TrialNumber.unique(), 20, 0.2)

    bins = np.arange(0, (PRE_INTERVAL + POST_INTERVAL) / 1000, 0.1)
    outputs = classifier_utils.evaluate_classifiers_by_time_bins(
        wrapped, inputs, labels, bins, random_splitter, cards=card_idxs
    )
    io_utils.save_model_outputs(
        "action", 
        f"{PRE_INTERVAL}_fb_{POST_INTERVAL}",
        "random",
        outputs,
        base_dir=base_dir
    )

def generate_figure(base_dir):
    """
    For each session, runs decoders, generates figures
    """
    _, test_accs_by_bin, _, _, _ = io_utils.load_model_outputs(
        "action", 
        f"{PRE_INTERVAL}_fb_{POST_INTERVAL}",
        "random",
        base_dir=base_dir
    )

    fig, ax = plt.subplots()
    visualization_utils.visualize_accuracy_across_time_bins(
        test_accs_by_bin,
        1.3, 1.5, 0.1,
        ax,
        label="Action",
        right_align=True, 
        # color='black'
    )

    ax.axvspan(-0.8, 0, alpha=0.3, color='gray')
    ax.axvline(0.098, alpha=0.3, color='gray', linestyle='dashed')
    ax.axhline(0.25, color='black', linestyle='dotted', label="Estimated Chance")
    ax.set_xlabel("Time Relative to Feedback (s)")
    ax.set_ylabel("Decoder Accuracy")
    ax.legend(prop={'size': 14})
    fig_dir = os.path.join(base_dir, "figures")
    if not os.path.isdir(fig_dir):
        os.mkdir(fig_dir)
    plt.savefig(os.path.join(fig_dir, f"action_decoding.png"))

def check_decoder_already_ran():
    # check 
    # TODO: can maybe implement later...
    return False

def check_figures_already_generated():
    # TODO: can maybe implement later...
    return False


def decode_features(row):
    sess_name = row.session_name
    print(f"Processing {sess_name}")

    base_dir = f"/data/patrick_scratch/multi_sess/{sess_name}/"
    # grab firing rates, behavioral data
    behavior_path = f"/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
    beh = pd.read_csv(behavior_path)
    valid_beh = beh[beh.Response.isin(["Correct", "Incorrect"])]
    card_idxs = behavioral_utils.get_shuffled_card_idxs(valid_beh)
    frs = pd.read_pickle(f"/data/patrick_scratch/multi_sess/{sess_name}/{sess_name}_firing_rates_{PRE_INTERVAL}_{EVENT}_{POST_INTERVAL}_{INTERVAL_SIZE}_bins.pickle")

    # run decoder
    # check if expected files all exist, if so, skip this session
    if not REPLACE and check_decoder_already_ran():
        print(f"Action Decoder for {sess_name} has already run, skipping")
    else: 
        print(f"Running Action Decoder for {sess_name}")
        run_decoder(frs, card_idxs, base_dir)
    # generate figures
    if not REPLACE and check_figures_already_generated():
        print(f"Action Figures for {sess_name} has already been generated, skipping")
    else: 
        print(f"Generating figure for {sess_name}")
        generate_figure(base_dir)

def main():
    valid_sess = pd.read_pickle("/data/patrick_scratch/multi_sess/valid_sessions.pickle")
    valid_sess.apply(decode_features, axis=1)

if __name__ == "__main__":
    main()