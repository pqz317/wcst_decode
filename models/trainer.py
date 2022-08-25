import torch
from torch import nn

class Trainer:
    def __init__(self, learning_rate=0.001, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        pass

    def train(self, model, x_train, y_train):
        x_train =  torch.Tensor(x_train)
        y_train = torch.Tensor(y_train).to(torch.long)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        model.train()

        for epoch_idx in range(self.max_iter):
            optimizer.zero_grad()
            out = model(x_train)
            loss = criterion(torch.squeeze(out), y_train)
            loss.backward()
            optimizer.step()