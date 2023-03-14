import torch
from torch import nn
import numpy as np
import copy
from torch.utils.data import DataLoader
import time


class Trainer:
    def __init__(self, learning_rate=0.005, max_iter=1000, batch_size=128, weight_decay=0.0, loss_fn=None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn

    def train(self, model, dataset, return_intermediates=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        # loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=3)
        # loader = DataLoader(dataset, batch_size=self.batch_size)
        if not self.loss_fn:
            self.loss_fn = nn.CrossEntropyLoss()
        criterion = self.loss_fn.to(device)
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay)
        model.train()

        losses = np.empty((self.max_iter))
        intermediates = np.empty((self.max_iter))

        for epoch_idx in range(self.max_iter):
            # x_train: num_training_trials x num_inputs (num_neurons)
            # for batch in loader:
            x_train, y_train, cards_train = dataset
            optimizer.zero_grad()
            if cards_train is not None: 
                out = model(x_train, cards_train)
            else:
                out = model(x_train)
            # print("out during training")
            # print(out[:10])
            # print("y_train")
            # print(y_train[:10])
            loss = criterion(torch.squeeze(out), y_train)
            loss.backward()
            optimizer.step()
            losses[epoch_idx] = loss.item()
            if return_intermediates:
                copied = copy.deepcopy(model)
                intermediates[epoch_idx] = copied
        model.eval()
        return losses