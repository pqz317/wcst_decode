import argparse
from single_selected_feature_configs import add_defaults_to_parser, SingleSelectedFeatureConfigs
from constants.behavioral_constants import *
from tqdm import tqdm
import decode_single_selected_features

NUM_SHUFFLES_PER_FEAT = 10

def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(SingleSelectedFeatureConfigs(), parser)
    args = parser.parse_args()

    for feat_idx in tqdm(range(len(FEATURES))):
        args.feat_idx = feat_idx
        decode_single_selected_features.main(args)

    for shuffle_idx in range(NUM_SHUFFLES_PER_FEAT):
        args.shuffle_idx = shuffle_idx
        for feat_idx in tqdm(range(len(FEATURES))):
            args.feat_idx = feat_idx
            decode_single_selected_features.main(args)

if __name__ == "__main__":
    main()