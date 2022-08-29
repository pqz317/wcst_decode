import torch
from torch import nn

class MultiLayer(nn.Module):
    """LogisticRegressor, single linear layer with sigmoid 

    Args:
        n_inputs (int): number of input units
        n_outputs (int): number of output units

    Attributes:
        in_layer (nn.Linear): weights and biases of input layer
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        super().__init__()  # needed to invoke the properties of the parent class nn.Module
        self.fc1 = nn.Linear(n_inputs, n_hidden) # neural activity --> output classes
        self.fc2 = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        y = torch.relu(self.fc2(h))
        return y