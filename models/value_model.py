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

    def forward(self, neural_activity, cards_by_trials):
        """
        Args
            neural_activity: batch_size x 59
            cards: batch_size x 4 (cards) x 3 features
        """

        feature_values = self.linear(neural_activity)     
        print(feature_values.shape)
        card_values = torch.empty((cards_by_trials.shape[0], cards_by_trials.shape[1]))
        for trial_idx in range(cards_by_trials.shape[0]):
            for card_idx in range(cards_by_trials.shape[1]):
                feature_idxs = cards_by_trials[trial_idx, card_idx]
                card_values[trial_idx, card_idx] = torch.sum(feature_values[trial_idx, feature_idxs])
        return card_values