"""
CMPUT 466/566 - Machine Learning, Winter 2024, Assignment 1
B. Chan

TODO: You will need to implement the following functions:
- train_nb: ndarray, ndarray, int -> Params
- predict_nb: Params, ndarray -> ndarray

Implementation description will be provided under each function.

For the following:
- N: Number of samples.
- D: Dimension of input features.
- C: Number of classes (labels). We assume the class starts from 0.
"""

import numpy as np


class Params:
    def __init__(self, means, covariances, priors, num_features, num_classes):
        """ This class represents the parameters of the Naive Bayes model,
            where the generative model is modeled as a Gaussian.
        NOTE: We assume lables are 0 to K - 1, where K is number of classes.

        We have three parameters to keep track of:
        - self.means (ndarray (shape: (K, D))): Mean for each of K Gaussian likelihoods.
        - self.covariances (ndarray (shape: (K, D, D))): Covariance for each of K Gaussian likelihoods.
        - self.priors (shape: (K, 1))): Prior probabilty of drawing samples from each of K class.

        Args:
        - num_features (int): The number of features in the input vector
        - num_classes (int): The number of classes in the task.
        """

        self.D = num_features
        self.C = num_classes

        # Shape: K x D
        self.means = means

        # Shape: K x D x D
        self.covariances = covariances

        # Shape: K x 1
        self.priors = priors

        assert self.means.shape == (self.C, self.D), f"means shape mismatch. Expected: {(self.C, self.D)}. Got: {self.means.shape}"
        assert self.covariances.shape == (self.C, self.D, self.D), f"covariances shape mismatch. Expected: {(self.C, self.D, self.D)}. Got: {self.covariances.shape}"
        assert self.priors.shape == (self.C, 1), f"priors shape mismatch. Expected: {(self.C, 1)}. Got: {self.priors.shape}"


def train_nb(train_X, train_y, num_classes, **kwargs):
    """ This trains the parameters of the NB model, given training data.

    Args:
    - train_X (ndarray (shape: (N, D))): NxD matrix storing N D-dimensional training inputs.
    - train_y (ndarray (shape: (N, 1))): Column vector with N scalar training outputs (labels).

    Output:
    - params (Params): The parameters of the NB model.
    """
    assert len(train_X.shape) == len(train_y.shape) == 2, f"Input/output pairs must be 2D-arrays. train_X: {train_X.shape}, train_y: {train_y.shape}"
    (N, D) = train_X.shape
    assert train_y.shape[1] == 1, f"train_Y must be a column vector. Got: {train_y.shape}"

    # Shape: C x D
    means = np.zeros((num_classes, D))

    # Shape: C x D x D
    covariances = np.tile(np.eye(D), reps=(num_classes, 1, 1))

    # Shape: C x 1
    priors = np.ones(shape=(num_classes, 1)) / num_classes

    # ====================================================
    # TODO: Implement your solution within the box
   
    # this is Guassian NB
    # train_nb fits the NB model using the Maximum Likelihood estimates.
    # For each class, compute the prior, and the mean and covariance of the Gaussian likelihood.

    # prior is based on the frequency of a class in the dataset - easy: # of occurence of class / N

    for c in range(num_classes):
        class_samples = train_X[train_y.flatten() == c]

        # Calculate mean
        means[c] = np.mean(class_samples, axis=0)

        for i in range(D):
            covariances[c, i, i] = np.var(class_samples[:, i])
        
        # Calculate covariance
        #covariances[c] = np.cov(class_samples, rowvar=False)

        # Calculate prior
        priors[c] = len(class_samples) / N

    
    # mean is the average value of the feature for a particular class - take the element-wise average of X


    # covariance? initialized as num_classes number of DxD identify matrix, can capture the covariance between every pair of features for each class

    # ====================================================

    params = Params(means, covariances, priors, D, num_classes)
    return params


def predict_nb(params, X):
    """ This function predicts the probability of labels given X.

    Args:
    - params (Params): The parameters of the NB model.
    - X (ndarray (shape: (N, D))): NxD matrix with N D-dimensional inputs.

    Output:
    - probs (ndarray (shape: (N, K))): NxK matrix storing N K-vectors (i.e. the K class probabilities)
    """
    assert len(X.shape) == 2, f"Input/output pairs must be 2D-arrays. X: {X.shape}"
    (N, D) = X.shape

    probs = np.zeros((N, params.C))
    unnormalized_probs = np.zeros((N, params.C))
    # ====================================================
    # TODO: Implement your solution within the box
    for c in range(params.C):
        # Calculate log-likelihood for each class

        # still need to be fixed
        log_likelihood = -0.5 * np.sum(np.log((2 * np.pi)**D * np.linalg.det(params.covariances[c, :, :]))) \
                         - 0.5 * np.sum(((X - params.means[c, :]) @ np.linalg.inv(params.covariances[c, :, :])) * (X - params.means[c, :]), axis=1)
        
        # Calculate unnormalized probability (log prior + log likelihood)
        unnormalized_probs[:, c] = np.log(params.priors[c]) + log_likelihood

    # Calculate normalized probabilities
    log_probs = unnormalized_probs - np.max(unnormalized_probs, axis=1, keepdims=True)  # for numerical stability
    probs = np.exp(log_probs)
    probs /= np.sum(probs, axis=1, keepdims=True)
    # ====================================================

    return probs
