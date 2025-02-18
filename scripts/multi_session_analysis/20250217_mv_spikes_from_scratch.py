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


SUBJECT = "BL"
SCRATCH_DIR = "/data/patrick_res/scratch"

def main():
    sess_dir_names = os.listdir(SCRATCH_DIR)
    for sess_dir_name in sess_dir_names:
        print(f"copying for session {sess_dir_name}")
        dest_dir = f"/data/rawdata/sub-{SUBJECT}/{sess_dir_name}/spikes"
        os.makedirs(dest_dir, exist_ok=True)
        src_dir = f"{SCRATCH_DIR}/{sess_dir_name}/spikes"
        file_names = os.listdir(src_dir)
        for file_name in file_names: 
            full_file_path = os.path.join(src_dir, file_name)
            # print(full_file_path)
            shutil.copy(full_file_path, dest_dir)

if __name__ == "__main__":
    main()