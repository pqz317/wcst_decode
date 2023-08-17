import torch
from torch import nn
import numpy as np
import copy
from torch.utils.data import DataLoader
import time


class Trainer:
    def __init__(
        self, 
        learning_rate=0.005, 
        max_iter=1000, 
        batch_size=128, 
        weight_decay=0.0, 
        loss_fn=None,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn

    def train(self, model, dataset, valid_dataset=None):
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
        loss_dict = {}
        loss_dict["train_losses"] = []
        loss_dict["valid_losses"] = []

        for epoch_idx in range(self.max_iter):
            # x_train: num_training_trials x num_inputs (num_neurons)
            # for batch in loader:
            optimizer.zero_grad()
            x_train, y_train, cards_train = dataset
            if cards_train is not None: 
                out = model(x_train, cards_train)
            else:
                out = model(x_train)
            loss = criterion(torch.squeeze(out), y_train)

            loss.backward()
            optimizer.step()
            loss_dict["train_losses"].append(loss.item())

            if valid_dataset:
                with torch.no_grad():
                    model.eval()
                    x_valid, y_valid, card_valid = valid_dataset
                    if cards_train is not None: 
                        out = model(x_valid, card_valid)
                    else:
                        out = model(x_valid)
                    valid_loss = criterion(torch.squeeze(out), y_valid)
                    loss_dict["valid_losses"].append(valid_loss.item())
                    model.train()
        model.eval()
        return loss_dict