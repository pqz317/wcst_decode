import numpy as np
import torch

class ModelWrapper:
    def __init__(self, model_type, init_params, trainer, labels):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type
        self.init_params = init_params
        self.trainer = trainer
        self.labels_to_idx = {label: idx for idx, label in enumerate(labels)}
        self.idx_to_labels = {idx: label for idx, label in enumerate(labels)}

    def fit(self, x_train, y_train, cards_train=None):
        self.model = self.model_type(**self.init_params)
        y_train_idxs = np.array([self.labels_to_idx[label] for label in y_train.tolist()]).astype(int)
        self.trainer.train(self.model, x_train, y_train_idxs, cards_train)
        return self

    def predict(self, x_test, cards_test=None):
        x_test = torch.Tensor(x_test).to(self.device)
        cards_test = torch.Tensor(cards_test).to(torch.long).to(self.device)
        if cards_test is not None:
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