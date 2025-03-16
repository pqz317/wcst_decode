import numpy as np
import torch
from models.wcst_dataset import WcstDataset

class ModelWrapper:
    """
    A wrapper class for machine learning models that handles training, prediction, and evaluation.
    
    This class provides a consistent interface for working with different types of models,
    handling data conversion between numpy arrays and PyTorch tensors, and managing device placement.

    Parameters
    ----------
    model_type : class
        The class of the model to be instantiated
    init_params : dict
        Dictionary of initialization parameters for the model
    trainer : object
        Trainer object that handles the training process
    labels : list
        List of unique labels used for classification
    """

    def __init__(self, model_type, init_params, trainer, labels):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        self.model_type = model_type
        self.init_params = init_params
        self.trainer = trainer
        self.labels_to_idx = {label: idx for idx, label in enumerate(labels)}
        self.idx_to_labels = {idx: label for idx, label in enumerate(labels)}
    
    def _create_dataset(self, x, y, cards=None):
        """
        Creates a dataset from input features, labels, and optional card information.

        Parameters
        ----------
        x : numpy.ndarray
            Input features
        y : numpy.ndarray
            Target labels
        cards : numpy.ndarray, optional
            Additional card information

        Returns
        -------
        tuple
            Tuple containing (x_tensor, y_tensor, cards_tensor) all moved to appropriate device
        """
        y_idxs = np.array([self.labels_to_idx[label] for label in y.tolist()]).astype(int)
        # dataset = WcstDataset(x_train, y_train_idxs, cards_train)
        ### TEST CODE
        x = torch.tensor(x).float().to(self.device)
        y = torch.tensor(y_idxs).to(self.device)
        if cards is not None:
            cards = torch.tensor(cards).to(torch.long).to(self.device)
        return (x, y, cards)

    def fit(self, x_train, y_train, cards_train=None):
        """
        Fits the model to the training data.

        Parameters
        ----------
        x_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training labels
        cards_train : numpy.ndarray, optional
            Training card information

        Returns
        -------
        self
            Returns self for method chaining
        """
        self.model = self.model_type(**self.init_params)
        dataset = self._create_dataset(x_train, y_train, cards_train)
        self.trainer.train(self.model, dataset)
        return self

    def fit_with_valid(self, x_train, y_train, x_valid, y_valid, cards_train=None, cards_valid=None):
        """
        Fits the model using both training and validation data.

        Parameters
        ----------
        x_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training labels
        x_valid : numpy.ndarray
            Validation features
        y_valid : numpy.ndarray
            Validation labels
        cards_train : numpy.ndarray, optional
            Training card information
        cards_valid : numpy.ndarray, optional
            Validation card information

        Returns
        -------
        dict
            Training history and metrics
        """
        self.model = self.model_type(**self.init_params)
        train_dataset = self._create_dataset(x_train, y_train, cards_train)
        valid_dataset = self._create_dataset(x_valid, y_valid, cards_valid)
        return self.trainer.train(self.model, train_dataset, valid_dataset)

    def predict(self, x_test, cards_test=None):
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
        self.model.eval()
        x_test = torch.Tensor(x_test).to(self.device)
        if cards_test is not None:
            cards_test = torch.Tensor(cards_test).to(torch.long).to(self.device)
            probs = self.model(x_test, cards_test)
        else:
            probs = self.model(x_test)
        label_idxs = np.argmax(probs.detach().cpu().numpy(), axis=1)
        return np.array([self.idx_to_labels[label_idxs] for label_idxs in label_idxs.tolist()])

    def score(self, x_test, y_test, cards_test=None):
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
        labels = self.predict(x_test, cards_test)
        score = np.sum(labels == y_test) / len(y_test)
        return score

    @property
    def coef_(self):
        """
        Returns the model's coefficients/weights.

        Returns
        -------
        numpy.ndarray
            Model coefficients
        """
        return self.model.linear.weight.detach().cpu().numpy()
    
class ModelWrapperLinearRegression: 
    """
    A wrapper where score and predictions are for a Linear instead of Logistic Regression
    """

    def fit(self, x_train, y_train, cards_train=None):   
        ones = np.ones((x_train.shape[0], 1))
        x_train_ones = np.hstack((ones, x_train))
        self.weights = np.linalg.pinv(x_train_ones) @ y_train
        return self

    def predict(self, x_test, cards_test=None):
        ones = np.ones((x_test.shape[0], 1))
        x_test_ones = np.hstack((ones, x_test))
        return x_test_ones @ self.weights

    def score(self, x_test, y_test, cards_test=None):
        outs = self.predict(x_test, cards_test)
        # print(f"outs: {outs[:10]}")
        # print(f"ys: {y_test[:10]}")
        score = np.sum((outs - y_test)**2) / len(y_test)
        # print(f"score: {score}")
        return score

    # @property
    # def coef_(self):
    #     return self.model.linear.weight.detach().cpu().numpy()

class ModelWrapperRegression: 
    """
    A wrapper class specifically for regression models that handles training and evaluation.
    
    This class provides functionality for regression tasks, including custom scoring metrics
    like pseudo R-squared calculations.

    Parameters
    ----------
    model_type : class
        The class of the regression model to be instantiated
    init_params : dict
        Dictionary of initialization parameters for the model
    trainer : object
        Trainer object that handles the training process
    """

    def __init__(self, model_type, init_params, trainer):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # print(self.device)
        self.model_type = model_type
        self.init_params = init_params
        self.trainer = trainer

    def _create_dataset(self, x, y):
        """
        Creates a dataset from input features and target values.

        Parameters
        ----------
        x : numpy.ndarray
            Input features
        y : numpy.ndarray
            Target values

        Returns
        -------
        tuple
            Tuple containing (x_tensor, y_tensor, None) moved to appropriate device
        """
        x = torch.tensor(x).float().to(self.device)
        y = torch.tensor(y).to(self.device)
        return (x, y, None)

    def fit(self, x_train, y_train):
        """
        Fits the regression model to the training data.

        Parameters
        ----------
        x_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training target values

        Returns
        -------
        self
            Returns self for method chaining
        """
        self.model = self.model_type(**self.init_params)
        dataset = self._create_dataset(x_train, y_train)
        self.trainer.train(self.model, dataset)
        return self

    def predict(self, x_test):
        """
        Makes predictions on test data.

        Parameters
        ----------
        x_test : numpy.ndarray
            Test features

        Returns
        -------
        numpy.ndarray
            Predicted values
        """
        self.model.eval()
        x_test = torch.Tensor(x_test).to(self.device)
        res = self.model(x_test)
        return res.detach().cpu().numpy()

    def score(self, x_test, y_test):
        """
        Calculates the pseudo R-squared score for the model predictions.
        
        The score is calculated as (L_model - L_null) / (L_saturated - L_null),
        where L_null is the likelihood using only the mean for prediction,
        and L_saturated is NLL(y_true, y_true).

        Parameters
        ----------
        x_test : numpy.ndarray
            Test features
        y_test : numpy.ndarray
            True target values

        Returns
        -------
        float
            Pseudo R-squared score
        """
        loss_fn = self.trainer.loss_fn.to(self.device)
        x, y_true, _ = self._create_dataset(x_test, y_test)
        y_pred = self.model(x)
        y_null = torch.mean(y_true).repeat(len(y_true))
        l_model = -1* loss_fn(y_pred, y_true)
        y_null[y_null == 0] = 1e8
        l_null = -1 * loss_fn(torch.log(y_null), y_true)
        y_true[y_true == 0] = 1e8
        l_sat = -1 * loss_fn(torch.log(y_true), y_true)
        return ((l_model - l_null) / (l_sat - l_null)).detach().cpu().numpy()

    @property
    def coef_(self):
        """
        Returns the model's coefficients/weights.

        Returns
        -------
        numpy.ndarray
            Model coefficients
        """
        return self.model.linear.weight.detach().cpu().numpy()
    
class ModelWrapperLinearRegression:
    """
    A wrapper class for simple linear regression using numpy's linear algebra operations.
    
    This class implements basic linear regression functionality without using PyTorch,
    suitable for simple regression tasks.
    """

    def fit(self, x_train, y_train, cards_train=None):
        """
        Fits a linear regression model using the normal equation.

        Parameters
        ----------
        x_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training target values
        cards_train : numpy.ndarray, optional
            Ignored parameter for API compatibility

        Returns
        -------
        self
            Returns self for method chaining
        """
        ones = np.ones((x_train.shape[0], 1))
        x_train_ones = np.hstack((ones, x_train))
        self.weights = np.linalg.pinv(x_train_ones) @ y_train
        return self

    def predict(self, x_test, cards_test=None):
        """
        Makes predictions using the fitted linear regression model.

        Parameters
        ----------
        x_test : numpy.ndarray
            Test features
        cards_test : numpy.ndarray, optional
            Ignored parameter for API compatibility

        Returns
        -------
        numpy.ndarray
            Predicted values
        """
        ones = np.ones((x_test.shape[0], 1))
        x_test_ones = np.hstack((ones, x_test))
        return x_test_ones @ self.weights

    def score(self, x_test, y_test, cards_test=None):
        """
        Calculates the mean squared error score for the predictions.

        Parameters
        ----------
        x_test : numpy.ndarray
            Test features
        y_test : numpy.ndarray
            True target values
        cards_test : numpy.ndarray, optional
            Ignored parameter for API compatibility

        Returns
        -------
        float
            Mean squared error score
        """
        outs = self.predict(x_test, cards_test)
        # print(f"outs: {outs[:10]}")
        # print(f"ys: {y_test[:10]}")
        score = np.sum((outs - y_test)**2) / len(y_test)
        # print(f"score: {score}")
        return score

    # @property
    # def coef_(self):
    #     return self.model.linear.weight.detach().cpu().numpy()

class DummyCombinedWrapper: 
    """
    A wrapper class that combines multiple models into a single interface.
    
    This class is useful for ensemble methods or when multiple models need to be
    treated as a single unit.

    Parameters
    ----------
    models : list
        List of model objects to be combined
    """
    def __init__(self, models):
        self.models = models

    @property
    def coef_(self):
        """
        Returns the combined coefficients from all models.

        Returns
        -------
        numpy.ndarray
            Stacked array of coefficients from all models
        """
        weights = []
        for model in self.models:
            weights.append(model.coef_)
        return np.vstack(weights)

    @staticmethod
    def combine_models(model_list):
        """
        Creates a combined model wrapper from a list of models.

        Parameters
        ----------
        model_list : list
            List of models, each as an np array with shape (num_time_bins x num_splits)

        Returns
        -------
        numpy.ndarray
            Array of combined model wrappers with shape (num_time_bins x num_splits)
        """
        num_time_bins, num_splits = model_list[0].shape
        combined = np.empty((num_time_bins, num_splits), dtype=object)
        for time_idx in range(num_time_bins):
            for split_idx in range(num_splits):
                models = [m[time_idx, split_idx] for m in model_list]
                combined[time_idx, split_idx] = DummyCombinedWrapper(models)
        return combined
        