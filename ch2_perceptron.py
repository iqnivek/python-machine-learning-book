import numpy as np

class Perceptron(object):
    """
    eta - learning rate from 0.0 to 1.0
    n_iter - num iterations
    random_state - RNG seed for random weight initialization
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        X - [n_samples, n_features] dimensional training matrix
        y - [n_samples, 1] vector, target values

        weights_ - weights after fitting
        errors_ - number of misclassifications in each epoch
        """

        rgen = np.random.RandomState(self.random_state)
        # initialize weights to small random numbers drawn from a
        # normal distribution with stddev 0.01
        self.weights_ = rgen.normal(
            loc=0.0,
            scale=0.01,
            size=1 + X.shape[1]
        )
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.weights_[1:] += update * xi
                self.weights_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        # calculate W_transpose * x
        return np.dot(X, self.weights_[1:]) + self.weights_[0]

    def predict(self, X):
        """ Return class label after unit step """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
