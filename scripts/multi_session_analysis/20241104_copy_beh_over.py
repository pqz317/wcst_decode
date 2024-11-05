import os
import shutil
import pandas as pd

"""
In an effort to consolidate and reorganize data, copy all object_features.csv to /data/patrick_res/behavior. 
Rename by subject
"""
subjects = ["SA", "BL"]

OLD_PATH = "/data/rawdata/sub-{sub}/sess-{sess_name}/behavior/sub-{sub}_sess-{sess_name}_object_features.csv"
NEW_PATH = "/data/patrick_res/behavior/{sub}/{sess_name}_object_features.csv"

def main():
    # handle 
    def copy_beh(sub, sess_name): 
        old_path = OLD_PATH.format(
            sub=sub, 
            sess_name=sess_name,
        )
        new_path = NEW_PATH.format(
            sub=sub,
            sess_name=sess_name,
        )
        shutil.copyfile(old_path, new_path)
        
    for sub in subjects: 
        sessions = pd.read_pickle(f"/data/patrick_res/sessions/{sub}/valid_sessions.pickle")
        sessions.apply(lambda x: copy_beh(sub, x.session_name), axis=1)

if __name__ == "__main__":
    main()