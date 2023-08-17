import torch
from torch import nn

class MultinomialLogisticRegressor(nn.Module):
    """LogisticRegressor, single linear layer 

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


class NormedMultinomialLogisticRegressor(nn.Module):
    """LogisticRegressor, batched normed inputs, single linear layer

    Args:
        n_inputs (int): number of input units
        n_outputs (int): number of output units

    Attributes:
        in_layer (nn.Linear): weights and biases of input layer
    """

    def __init__(self, n_inputs, n_classes):
        super().__init__()  # needed to invoke the properties of the parent class nn.Module
        self.norm = nn.BatchNorm1d(n_inputs, affine=False)
        self.linear = nn.Linear(n_inputs, n_classes) # neural activity --> output classes

    def forward(self, x):
        return self.linear(self.norm(x))

class NormedDropoutMultinomialLogisticRegressor(nn.Module):
    """LogisticRegressor, batched normed inputs, single linear layer

    Args:
        n_inputs (int): number of input units
        n_outputs (int): number of output units

    Attributes:
        in_layer (nn.Linear): weights and biases of input layer
    """

    def __init__(self, n_inputs, n_classes, p_dropout):
        super().__init__()  # needed to invoke the properties of the parent class nn.Module
        self.norm = nn.BatchNorm1d(n_inputs, affine=False)
        self.dropout = nn.Dropout(p=p_dropout)
        self.linear = nn.Linear(n_inputs, n_classes) # neural activity --> output classes

    def forward(self, x):
        return self.linear(self.dropout(self.norm(x)))

class NormedDropoutNonlinear(nn.Module):
    """LogisticRegressor, batched normed inputs, single linear layer

    Args:
        n_inputs (int): number of input units
        n_outputs (int): number of output units

    Attributes:
        in_layer (nn.Linear): weights and biases of input layer
    """

    def __init__(self, n_inputs, n_classes, p_dropout, hidden_sizes=[200]):
        super().__init__()  # needed to invoke the properties of the parent class nn.Module
        self.norm = nn.BatchNorm1d(n_inputs, affine=False)
        self.dropout = nn.Dropout(p=p_dropout)
        sequence = [self.norm, self.dropout]
        prev_dim = n_inputs
        for hidden in hidden_sizes:
            layer = nn.Linear(prev_dim, hidden)
            sequence.append(layer)
            sequence.append(nn.ReLU())
            prev_dim = hidden
        self.output = nn.Linear(prev_dim, n_classes) # neural activity --> output classes
        sequence.append(self.output)
        self.sequence = nn.Sequential(*sequence)

    def forward(self, x):
        return self.sequence(x)