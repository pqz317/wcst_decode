import numpy as np
import torch
from models.wcst_dataset import WcstDataset
import torch.nn.functional as F
from constants.decoding_constants import *

MODE_TO_CHOICE_REWARD_LABELS = {
    "chose_and_correct": [["Chose", "Correct"]],
    "updates_beliefs": [["Chose", "Correct"], ["Not Chose", "Incorrect"]],
} 

class ChoiceRewardModel:
    """
    Model that predicts interactions of choice/reward by using outputs 
    of a choice model and a reward model separately

    Parameters
    ----------
    choice_model
    reward_model

    """

    def __init__(self, choice_model, reward_model, mode):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        choice_model.model.to(self.device)
        reward_model.model.to(self.device)
        self.choice_model = choice_model
        self.reward_model = reward_model
        self.mode = mode
        self.classes = MODE_TO_CLASSES[mode]
        class_mask = torch.zeros((2, 2))
        for choice_label, reward_label in MODE_TO_CHOICE_REWARD_LABELS[mode]:
            choice_idx = self.choice_model.labels_to_idx[choice_label]
            reward_idx = self.reward_model.labels_to_idx[reward_label]
            class_mask[choice_idx, reward_idx] = 1
        self.class_mask = class_mask.to(self.device)

    def predict(self, x_test):
        """
        Makes predictions on test data.

        Parameters
        ----------
        x_test : numpy.ndarray
            Test features
        cards_test : numpy.ndarray, optional
            Test card information

        Returns
        -------
        numpy.ndarray
            Predicted labels
        """
        x_test = torch.Tensor(x_test).to(self.device)
        # want: two probs of N x 2
        choice_probs = F.softmax(self.choice_model.model(x_test), dim=1)
        reward_probs = F.softmax(self.reward_model.model(x_test), dim=1)

        # choice_probs = torch.ones((x_test.shape[0], 2)) * 0.5
        # reward_probs = torch.ones((x_test.shape[0], 2)) * 0.5

        # makes choice probs N x 2 x 1, reward_probs N x 1 x 2, matmuls 
        # int probs: N x 2 x 2
        int_probs = choice_probs.unsqueeze(2) @ reward_probs.unsqueeze(1)

        # element-wise multiply interaction probabilities with mask, sum per row. 
        pos_probs = (int_probs * self.class_mask).sum(axis=(1, 2))
        labels = np.where(pos_probs.detach().cpu().numpy() > 0.5, self.classes[0], self.classes[1])
        return labels

    def score(self, x_test, y_test):
        """
        Calculates the accuracy score for the model predictions.

        Parameters
        ----------
        x_test : numpy.ndarray
            Test features
        y_test : numpy.ndarray
            True labels
        cards_test : numpy.ndarray, optional
            Test card information

        Returns
        -------
        float
            Accuracy score between 0 and 1
        """
        labels = self.predict(x_test)
        score = np.sum(labels == y_test) / len(y_test)
        return score
    
def create_models(choice_models, reward_models, mode):
    """
    Grabs choice and reward models, of shape [n_time_bins, n_runs], 
    """
    def create_model(choice_model, reward_model):
        return ChoiceRewardModel(choice_model, reward_model, mode)
    return np.vectorize(create_model)(choice_models, reward_models)