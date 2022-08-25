import torch
from torch import nn

class MultinomialLogisticRegressor(nn.Module):
    """LogisticRegressor, single linear layer with sigmoid 

    Args:
        n_inputs (int): number of input units
        n_outputs (int): number of output units

    Attributes:
        in_layer (nn.Linear): weights and biases of input layer
    """

    def __init__(self, n_inputs, n_classes):
        super().__init__()  # needed to invoke the properties of the parent class nn.Module
        self.linear = nn.Linear(n_inputs, n_classes) # neural activity --> output classes

    def forward(self, x):
        return self.linear(x)