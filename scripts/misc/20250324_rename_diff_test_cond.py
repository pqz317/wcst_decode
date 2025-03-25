import os

def main():
    dir = "/data/patrick_res/single_selected_diff_test_cond/SA_FeedbackOnsetLong_v2_pseudo/shuffles"
    for name in os.listdir(dir):
        parts = name.split("_")
        feat = parts[0]
        shuffle_idx = parts[3]
        cond = parts[4]
        if cond == "not":
            cond = "not_pref"
        new_name = f"{feat}_{cond}_shuffle_{shuffle_idx}_test_accs.npy"
        os.rename(os.path.join(dir, name), os.path.join(dir, new_name))

if __name__ == "__main__":
    main()