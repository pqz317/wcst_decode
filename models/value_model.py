import torch
from torch import nn

class ValueModel(nn.Module):
    """LogisticRegressor, single linear layer with sigmoid 

    Args:
        n_inputs (int): number of input units
        n_outputs (int): number of output units

    Attributes:
        in_layer (nn.Linear): weights and biases of input layer
    """

    def __init__(self, n_inputs, n_values):
        super().__init__()  # needed to invoke the properties of the parent class nn.Module
        self.linear = nn.Linear(n_inputs, n_values) # neural activity --> output classes

    def forward(self, neural_activity, card_masks):
        """
        Args
            neural_activity: batch_size x 59
            card_masks: batch_size x 4 (cards) x 12 features
        """

        # batch_size x 12
        feature_values = self.linear(neural_activity)     

        # add dimension, repeat along added dim
        # new dims batch_size x 4 x 12
        expanded = feature_values.unsqueeze(1).repeat(1, 4, 1)
        masked = expanded * card_masks
        summed = torch.sum(masked, dim=2)
        return summed