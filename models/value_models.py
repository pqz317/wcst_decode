import torch
from torch import nn


class ExpModule(nn.Module):
    """
    Exponential function element-wise
    """

    def __init__(self):
        super(ExpModule, self).__init__()

    def forward(self, input):
        return torch.exp(input)

class FeatureValueBaseModel(nn.Module):
    """Base class for any feature-value models

    Args: 
        agg_func: function for how to aggregate values of features into values of card
    """

    def __init__(self, agg_func=torch.sum):
        super().__init__()  # needed to invoke the properties of the parent class nn.Module
        self.agg_func = agg_func

    def choice_from_values(self, feature_values, card_masks):
        # add dimension, repeat along added dim
        # new dims batch_size x 4 x 12
        expanded = feature_values.unsqueeze(1).repeat(1, 4, 1)
        masked = expanded * card_masks
        values_agg = self.agg_func(masked, dim=2)
        if self.agg_func == torch.max:
            # annoying check because max func returns both values and indices
            values_agg = values_agg.values
        return values_agg       


class ValueLinearModel(FeatureValueBaseModel):
    """Model where neural activity linearly maps to feature values

    Args:
        n_inputs (int): number of input units
        n_values (int): number of feature values
    """

    def __init__(self, n_inputs, n_values, agg_func=torch.sum):
        super().__init__(agg_func)
        self.linear = nn.Linear(n_inputs, n_values) # neural activity --> output classes

    def forward(self, neural_activity, card_masks):
        """
        Args
            neural_activity: batch_size x 59
            card_masks: batch_size x 4 (cards) x 12 features
        """
        # batch_size x 12
        feature_values = self.linear(neural_activity)     
        return self.choice_from_values(feature_values, card_masks)


class ValueReLUModel(FeatureValueBaseModel):
    """Model where feature values are derived from a linearly mapped neural activity
    after a ReLU function

    Args:
        n_inputs (int): number of input units
        n_values (int): number of feature values
    """

    def __init__(self, n_inputs, n_values, agg_func=torch.sum):
        super().__init__(agg_func) 
        self.linear = nn.Linear(n_inputs, n_values) # neural activity --> output classes
        self.relu = nn.ReLU()

    def forward(self, neural_activity, card_masks):
        """
        Args
            neural_activity: batch_size x 59
            card_masks: batch_size x 4 (cards) x 12 features
        """

        # batch_size x 12
        linear_out = self.linear(neural_activity)   
        feature_values = self.relu(linear_out)  
        return self.choice_from_values(feature_values, card_masks)


class ValueMultilayerModel(FeatureValueBaseModel):
    """Model where neural activity goes through a hidden layer 

    Args:
        n_inputs (int): number of input units
        n_values (int): number of feature values
    """

    def __init__(self, n_inputs, n_hidden, n_values, agg_func=torch.sum):
        super().__init__(agg_func) 
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_values)
        self.relu2 = nn.ReLU()

    def forward(self, neural_activity, card_masks):
        """
        Args
            neural_activity: batch_size x 59
            card_masks: batch_size x 4 (cards) x 12 features
        """

        # batch_size x 12
        x = self.relu1(self.fc1(neural_activity))
        feature_values = self.relu2(self.fc2(x))
        return self.choice_from_values(feature_values, card_masks) 


class ValueExpModel(FeatureValueBaseModel):
    """Model where neural activity linearly maps to feature values
    Then feature values go through exponential (for interpretability)
    feature values map to card values then card values go through log 
    (to avoid redundant exp in cross-entropy)

    Args:
        n_inputs (int): number of input units
        n_values (int): number of feature values
    """

    def __init__(self, n_inputs, n_values, agg_func=torch.sum):
        super().__init__(agg_func)
        self.linear = nn.Linear(n_inputs, n_values) # neural activity --> output classes
        self.exp = ExpModule()

    def forward(self, neural_activity, card_masks):
        """
        Args
            neural_activity: batch_size x 59
            card_masks: batch_size x 4 (cards) x 12 features
        """
        # batch_size x 12
        feature_logits = self.linear(neural_activity)     
        feature_probs = self.exp(feature_logits)
        choice_probs = self.choice_from_values(feature_probs, card_masks)
        return torch.log(choice_probs)

