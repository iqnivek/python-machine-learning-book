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


"""
>>> import pandas as pd
>>> df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
>>> df.tail()
       0    1    2    3               4
       145  6.7  3.0  5.2  2.3  Iris-virginica
       146  6.3  2.5  5.0  1.9  Iris-virginica
       147  6.5  3.0  5.2  2.0  Iris-virginica
       148  6.2  3.4  5.4  2.3  Iris-virginica
       149  5.9  3.0  5.1  1.8  Iris-virginica

>>> import matplotlib.pyplot as plt
>>> import numpy as np

>>> # select setosa and versicolor
... y = df.iloc[0:100, 4].values
>>> y = np.where(y == 'Iris-setosa', -1, 1)

>>> # extract sepal length and petal length
... X = df.iloc[0:100, [0, 2]].values
... plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
<matplotlib.collections.PathCollection object at 0x10ea7f438>
>>> plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
<matplotlib.collections.PathCollection object at 0x10ea7f860>
>>> plt.xlabel('sepal length (cm)')
Text(0.5,0,'sepal length (cm)')
>>> plt.ylabel('petal length (cm)')
Text(0,0.5,'petal length (cm)')
>>> plt.legend(loc='upper left')
<matplotlib.legend.Legend object at 0x107d0c7f0>
>>> plt.show()

>>> ppn = Perceptron(eta=0.1, n_iter=10)
>>> ppn.fit(X, y)
<ch2_perceptron.Perceptron object at 0x10cee6278>
>>> plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
[<matplotlib.lines.Line2D object at 0x1154d5198>]
>>> plt.xlabel('Epochs')
Text(0.5,0,'Epochs')
>>> plt.ylabel('Number of updates')
Text(0,0.5,'Number of updates')
>>> plot.show()
"""
