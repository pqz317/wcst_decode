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


SCRATCH_DIR = "/data/patrick_res/scratch/object_features"

# example
# example correct path: "/data/rawdata/sub-BL/sess-20190123/behavior/sub-BL_sess-20190123_object_features.csv"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', default=True, type=lambda x: bool(strtobool(x)))
    parser.add_argument('--to_patrick_res', default=False, type=lambda x: bool(strtobool(x)))
    args = parser.parse_args()

    ofs = os.listdir(SCRATCH_DIR)
    for of_name in ofs:
        sub_str, sess_str, _, _ = of_name.split("_")
        sub = sub_str.split("-")[1]
        sess = sess_str.split("-")[1]
        print(sub_str)
        print(sess_str)
        if args.to_patrick_res:
            dest_dir = f"/data/patrick_res/behavior/{sub}"
            dest_name = f"{sess}_object_features.csv"
        else: 
            dest_dir = f"/data/rawdata/{sub_str}/{sess_str}/behavior"
            dest_name = of_name
        old_path = os.path.join(SCRATCH_DIR, of_name)
        dest_path = os.path.join(dest_dir, dest_name)
        print(f"Copying file {old_path} to {dest_path}")
        if not args.dry_run:
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(old_path, dest_path)

if __name__ == "__main__":
    main()