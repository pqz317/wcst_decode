# Script to move object features over from scratch folder, to correct location in /data/rawdata
import numpy as np
import pandas as pd
from spike_tools import (
    general as spike_general,
    analysis as spike_analysis,
)
import utils.behavioral_utils as behavioral_utils
import utils.spike_utils as spike_utils
import os
import argparse
import shutil
import argparse
from distutils.util import strtobool


# SCRATCH_DIR = "/data/patrick_res/scratch/sessioninfomodified"
# DEST_DIR = "/data/rawdata/sub-{sub}/sess-{sess}/session_info"

SCRATCH_DIR = "/data/patrick_res/scratch/sessioninfomodified_corrected"
DEST_DIR = "/data/rawdata/sub-{sub}/sess-{sess}/session_info_corrected"

# example session_info name: sub-BL_sess-20190129_sessioninfomodified.json
# example correct path: "/data/rawdata/sub-BL/sess-20190123/behavior/sub-BL_sess-20190123_object_features.csv"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', default=True, type=lambda x: bool(strtobool(x)))
    args = parser.parse_args()

    infos = os.listdir(SCRATCH_DIR)
    for info_name in infos:
        sub_str, sess_str, _ = info_name.split("_")
        sub = sub_str.split("-")[1]
        sess = sess_str.split("-")[1]
        print(sub_str)
        print(sess_str)
        dest_dir = DEST_DIR.format(sub=sub, sess=sess)
        old_path = os.path.join(SCRATCH_DIR, info_name)
        dest_path = os.path.join(dest_dir, info_name)
        print(f"Copying file {old_path} to {dest_path}")
        if not args.dry_run:
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(old_path, dest_path)

if __name__ == "__main__":
    main()