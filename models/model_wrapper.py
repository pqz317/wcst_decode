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

    def fit(self, x_train, y_train, cards_train=None):
        self.model = self.model_type(**self.init_params)
        y_train_idxs = np.array([self.labels_to_idx[label] for label in y_train.tolist()]).astype(int)
        # dataset = WcstDataset(x_train, y_train_idxs, cards_train)
        ### TEST CODE
        x_train = torch.tensor(x_train).float().to(self.device)
        y_train = torch.tensor(y_train_idxs).to(self.device)
        if cards_train is not None:
            cards_train = torch.tensor(cards_train).to(torch.long).to(self.device)
        dataset = (x_train, y_train, cards_train)
        self.trainer.train(self.model, dataset)
        return self

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