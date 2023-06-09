import s3fs
import os
import scipy.io
import numpy as np
import pickle
import torch

HUMAN_LFP_DIR = 'human-lfp'
NHP_DIR = 'nhp-lfp'
NHP_WCST_DIR = 'nhp-lfp/wcst-preprocessed/'

def get_fixation_times_path(subject, session):
    return os.path.join(
        NHP_WCST_DIR, "rawdata", f"sub-{subject}", 
        f"sess-{session}", "eye", "itemFixationTimes",
        f"sub-{subject}_sess-{session}_itemFixationTimes.m"
    )


def get_raw_fixation_times(fs, subject, session):
    """Grabs fixation times for each card displayed on the screen
    Note: per trial, each card can have multiple fixations. 

    Args:
        fs: Filesystem to grab from
        subject: str identifier for subject
        sesssion: which session to grab

    Returns:
        np.array of num trials length, with each element as a separate np array
            describing every fixation during the trial 
    """
    fixation_path = get_fixation_times_path(subject, session)
    with fs.open(fixation_path) as fixation_file:
        data = scipy.io.loadmat(fixation_file)
        return data["itemFixationTimes"]

# def save_model_outputs(fs, name, interval, split, outputs):
#     train_accs_by_bin, test_accs_by_bin, shuffled_accs, models, splits = outputs
#     np.save(fs.open(f"l2l.pqz317.scratch/{name}_train_accs_{interval}_{split}.npy", "wb"), train_accs_by_bin)
#     np.save(fs.open(f"l2l.pqz317.scratch/{name}_accs_{interval}_{split}.npy", "wb"), test_accs_by_bin)
#     np.save(fs.open(f"l2l.pqz317.scratch/{name}_shuffled_accs_{interval}_{split}.npy", "wb"), shuffled_accs)
#     np.save(fs.open(f"l2l.pqz317.scratch/{name}_models_{interval}_{split}.npy", "wb"), models)
#     pickle.dump(splits, fs.open(f"l2l.pqz317.scratch/{name}_splits_{interval}_{split}.npy", "wb"))

# def load_model_outputs(fs, name, interval, split):
#     train_accs_by_bin = np.load(fs.open(f"l2l.pqz317.scratch/{name}_train_accs_{interval}_{split}.npy", "rb"))
#     test_accs_by_bin = np.load(fs.open(f"l2l.pqz317.scratch/{name}_accs_{interval}_{split}.npy", "rb"))
#     shuffled_accs = np.load(fs.open(f"l2l.pqz317.scratch/{name}_shuffled_accs_{interval}_{split}.npy", "rb"))
#     models = np.load(fs.open(f"l2l.pqz317.scratch/{name}_models_{interval}_{split}.npy", "rb"), allow_pickle=True)
#     splits = pickle.load(fs.open(f"l2l.pqz317.scratch/{name}_splits_{interval}_{split}.npy", "rb"))
#     return train_accs_by_bin, test_accs_by_bin, shuffled_accs, models, splits

def save_model_outputs(name, interval, split, outputs):
    train_accs_by_bin, test_accs_by_bin, shuffled_accs, models, splits = outputs
    np.save(f"/data/patrick_scratch/{name}_train_accs_{interval}_{split}.npy", train_accs_by_bin)
    np.save(f"/data/patrick_scratch/{name}_accs_{interval}_{split}.npy", test_accs_by_bin)
    np.save(f"/data/patrick_scratch/{name}_shuffled_accs_{interval}_{split}.npy", shuffled_accs)
    np.save(f"/data/patrick_scratch/{name}_models_{interval}_{split}.npy", models)
    pickle.dump(splits, open(f"/data/patrick_scratch/{name}_splits_{interval}_{split}.npy", "wb"))

def load_model_outputs(name, interval, split):
    train_accs_by_bin = np.load(f"/data/patrick_scratch/{name}_train_accs_{interval}_{split}.npy")
    test_accs_by_bin = np.load(f"/data/patrick_scratch/{name}_accs_{interval}_{split}.npy")
    shuffled_accs = np.load(f"/data/patrick_scratch/{name}_shuffled_accs_{interval}_{split}.npy")
    models = np.load(f"/data/patrick_scratch/{name}_models_{interval}_{split}.npy", allow_pickle=True)
    splits = pickle.load(open(f"/data/patrick_scratch/{name}_splits_{interval}_{split}.npy", "rb"))
    return train_accs_by_bin, test_accs_by_bin, shuffled_accs, models, splits