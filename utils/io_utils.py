import s3fs
import os
import scipy.io

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
