import torch
from torch import nn
import numpy as np

class Trainer:
    def __init__(self, learning_rate=0.005, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        pass

    def train(self, model, x_train, y_train, cards_train=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        x_train = torch.Tensor(x_train).to(device)
        if cards_train is not None:
            cards_train = torch.Tensor(cards_train).to(torch.long).to(device)
        y_train = torch.Tensor(y_train).to(torch.long).to(device)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        model.train()

        losses = np.empty((self.max_iter))
        for epoch_idx in range(self.max_iter):
            # x_train: num_training_trials x num_inputs (num_neurons)
            optimizer.zero_grad()
            if cards_train is not None: 
                out = model(x_train, cards_train)
            else:
                out = model(x_train)
            loss = criterion(torch.squeeze(out), y_train)
            loss.backward()
            optimizer.step()
            losses[epoch_idx] = loss.item()
        return losses