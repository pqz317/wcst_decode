import numpy as np
import torch
from models.wcst_dataset import WcstDataset

class ModelWrapper:
    def __init__(self, model_type, init_params, trainer, labels):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        self.model_type = model_type
        self.init_params = init_params
        self.trainer = trainer
        self.labels_to_idx = {label: idx for idx, label in enumerate(labels)}
        self.idx_to_labels = {idx: label for idx, label in enumerate(labels)}
    
    def _create_dataset(self, x, y, cards=None):
        y_idxs = np.array([self.labels_to_idx[label] for label in y.tolist()]).astype(int)
        # dataset = WcstDataset(x_train, y_train_idxs, cards_train)
        ### TEST CODE
        x = torch.tensor(x).float().to(self.device)
        y = torch.tensor(y_idxs).to(self.device)
        if cards is not None:
            cards = torch.tensor(cards).to(torch.long).to(self.device)
        return (x, y, cards)

    def fit(self, x_train, y_train, cards_train=None):
        self.model = self.model_type(**self.init_params)
        dataset = self._create_dataset(x_train, y_train, cards_train)
        self.trainer.train(self.model, dataset)
        return self

    def fit_with_valid(self, x_train, y_train, x_valid, y_valid, cards_train=None, cards_valid=None):
        self.model = self.model_type(**self.init_params)
        train_dataset = self._create_dataset(x_train, y_train, cards_train)
        valid_dataset = self._create_dataset(x_valid, y_valid, cards_valid)
        return self.trainer.train(self.model, train_dataset, valid_dataset)

    def predict(self, x_test, cards_test=None):
        self.model.eval()
        x_test = torch.Tensor(x_test).to(self.device)
        if cards_test is not None:
            cards_test = torch.Tensor(cards_test).to(torch.long).to(self.device)
            probs = self.model(x_test, cards_test)
        else:
            probs = self.model(x_test)
        label_idxs = np.argmax(probs.detach().cpu().numpy(), axis=1)
        return np.array([self.idx_to_labels[label_idxs] for label_idxs in label_idxs.tolist()])

    def score(self, x_test, y_test, cards_test=None):
        labels = self.predict(x_test, cards_test)
        score = np.sum(labels == y_test) / len(y_test)
        return score

    @property
    def coef_(self):
        return self.model.linear.weight.detach().cpu().numpy()
    
class ModelWrapperLinearRegression: 
    """
    A wrapper where score and predictions are for a Linear instead of Logistic Regression
    """

    def fit(self, x_train, y_train, cards_train=None):   
        ones = np.ones((x_train.shape[0], 1))
        x_train_ones = np.hstack((ones, x_train))
        self.weights = np.linalg.pinv(x_train_ones) @ y_train
        return self

    def predict(self, x_test, cards_test=None):
        ones = np.ones((x_test.shape[0], 1))
        x_test_ones = np.hstack((ones, x_test))
        return x_test_ones @ self.weights

    def score(self, x_test, y_test, cards_test=None):
        outs = self.predict(x_test, cards_test)
        # print(f"outs: {outs[:10]}")
        # print(f"ys: {y_test[:10]}")
        score = np.sum((outs - y_test)**2) / len(y_test)
        # print(f"score: {score}")
        return score

    # @property
    # def coef_(self):
    #     return self.model.linear.weight.detach().cpu().numpy()

class ModelWrapperRegression: 
    """
    A wrapper where score and predictions are for a Linear instead of Logistic Regression
    """
    def __init__(self, model_type, init_params, trainer):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # print(self.device)
        self.model_type = model_type
        self.init_params = init_params
        self.trainer = trainer

    def _create_dataset(self, x, y):
        x = torch.tensor(x).float().to(self.device)
        y = torch.tensor(y).to(self.device)
        return (x, y, None)

    def fit(self, x_train, y_train):
        self.model = self.model_type(**self.init_params)
        dataset = self._create_dataset(x_train, y_train)
        self.trainer.train(self.model, dataset)
        return self

    def predict(self, x_test):
        self.model.eval()
        x_test = torch.Tensor(x_test).to(self.device)
        res = self.model(x_test)
        return res.detach().cpu().numpy()

    def score(self, x_test, y_test):
        """
        Report the pseudo R^2 
        (L_model - L_null) / (L_saturated - L_null)
        Where L_null is if only mean is used for prediction,
        L_saturated is NLL(y_true, y_true)
        """
        loss_fn = self.trainer.loss_fn.to(self.device)
        x, y_true, _ = self._create_dataset(x_test, y_test)
        y_pred = self.model(x)
        y_null = torch.mean(y_true).repeat(len(y_true))
        l_model = -1* loss_fn(y_pred, y_true)
        y_null[y_null == 0] = 1e8
        l_null = -1 * loss_fn(torch.log(y_null), y_true)
        y_true[y_true == 0] = 1e8
        l_sat = -1 * loss_fn(torch.log(y_true), y_true)
        return ((l_model - l_null) / (l_sat - l_null)).detach().cpu().numpy()

    @property
    def coef_(self):
        return self.model.linear.weight.detach().cpu().numpy()
    
class ModelWrapperLinearRegression: 
    """
    A wrapper where score and predictions are for a Linear instead of Logistic Regression
    """

    def fit(self, x_train, y_train, cards_train=None):   
        ones = np.ones((x_train.shape[0], 1))
        x_train_ones = np.hstack((ones, x_train))
        self.weights = np.linalg.pinv(x_train_ones) @ y_train
        return self

    def predict(self, x_test, cards_test=None):
        ones = np.ones((x_test.shape[0], 1))
        x_test_ones = np.hstack((ones, x_test))
        return x_test_ones @ self.weights

    def score(self, x_test, y_test, cards_test=None):
        outs = self.predict(x_test, cards_test)
        # print(f"outs: {outs[:10]}")
        # print(f"ys: {y_test[:10]}")
        score = np.sum((outs - y_test)**2) / len(y_test)
        # print(f"score: {score}")
        return score

    # @property
    # def coef_(self):
    #     return self.model.linear.weight.detach().cpu().numpy()

class DummyCombinedWrapper: 
    """
    A wrapper where score and predictions are for a Linear instead of Logistic Regression
    """
    def __init__(self, models):
        self.models = models

    @property
    def coef_(self):
        weights = []
        for model in self.models:
            weights.append(model.coef_)
        return np.vstack(weights)

    @staticmethod
    def combine_models(model_list):
        """
        Expect a list of models, each an np array num_time_bins x num_splits
        """
        num_time_bins, num_splits = model_list[0].shape
        combined = np.empty((num_time_bins, num_splits), dtype=object)
        for time_idx in range(num_time_bins):
            for split_idx in range(num_splits):
                models = [m[time_idx, split_idx] for m in model_list]
                combined[time_idx, split_idx] = DummyCombinedWrapper(models)
        return combined
        