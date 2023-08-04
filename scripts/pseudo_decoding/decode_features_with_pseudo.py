import os
import numpy as np
import pandas as pd
import utils.pseudo_utils as pseudo_utils
import utils.pseudo_classifier_utils as pseudo_classifier_utils

from utils.session_data import SessionData

from models.trainer import Trainer
from models.model_wrapper import ModelWrapper, ModelWrapperLinearRegression
from models.multinomial_logistic_regressor import NormedDropoutMultinomialLogisticRegressor



PRE_INTERVAL = 1300
POST_INTERVAL = 1500
INTERVAL_SIZE = 100
EVENT = "FeedbackOnset"

feature_dims = ["Color", "Shape", "Pattern"]
possible_features = {
    "Color": ['CYAN', 'GREEN', 'MAGENTA', 'YELLOW'],
    "Shape": ['CIRCLE', 'SQUARE', 'STAR', 'TRIANGLE'],
    "Pattern": ['ESCHER', 'POLKADOT', 'RIPPLE', 'SWIRL']
}

def decode_feature(feature_dim, valid_sess):
    print(f"Decoding {feature_dim}")
    sess_datas = valid_sess.apply(lambda x: SessionData.load_session_data(x.session_name, feature_dim), axis=1)
    classes = possible_features[feature_dim]
    num_neurons = sess_datas.apply(lambda x: x.get_num_neurons()).sum()
    init_params = {"n_inputs": num_neurons, "p_dropout": 0.5, "n_classes": len(classes)}
    trainer = Trainer(learning_rate=0.05, max_iter=1000, batch_size=1000)
    model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, classes)
    time_bins = np.arange(0, 2.8, 0.1)
    train_accs, test_accs, shuffled_accs, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(model, sess_datas, time_bins, 5, 2000, 400, 42)

    base_dir = "/data/patrick_scratch/pseudo"
    np.save(os.path.join(base_dir, f"{feature_dim}_train_accs.npy"), train_accs)
    np.save(os.path.join(base_dir, f"{feature_dim}_test_accs.npy"), test_accs)
    np.save(os.path.join(base_dir, f"{feature_dim}_shuffled_accs.npy"), shuffled_accs)
    np.save(os.path.join(base_dir, f"{feature_dim}_models.npy"), models)


def main():
    valid_sess = pd.read_pickle("/data/patrick_scratch/multi_sess/valid_sessions.pickle")
    for feature_dim in feature_dims: 
        decode_feature(feature_dim, valid_sess)

if __name__ == "__main__":
    main()