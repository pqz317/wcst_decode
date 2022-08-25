import numpy as np
import torch

class ModelWrapper:
    def __init__(self, model, trainer, labels):
        self.model = model
        self.trainer = trainer
        self.labels_to_idx = {label: idx for idx, label in enumerate(labels)}
        self.idx_to_labels = {idx: label for idx, label in enumerate(labels)}

    def fit(self, x_train, y_train):
        y_train_idxs = np.array([self.labels_to_idx[label] for label in y_train.tolist()]).astype(int)
        self.trainer.train(self.model, x_train, y_train_idxs)
        return self

    def predict(self, x_test):
        x_test =  torch.Tensor(x_test)
        probs = self.model(x_test)
        label_idxs = np.argmax(probs.detach().numpy(), axis=1)
        return np.array([self.idx_to_labels[label_idxs] for label_idxs in label_idxs.tolist()])

    def score(self, x_test, y_test):
        labels = self.predict(x_test)
        score = np.sum(labels == y_test) / len(y_test)
        return score

    @property
    def _coef(self):
        return self.model.linear.weight