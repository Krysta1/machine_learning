import numpy as np
import pandas as pd


class LogisticReg:
    def __init__(self, alpha=0.001, num_iter=5000):
        self.alpha = alpha
        self.num_iter = num_iter

    def x_plus_1(self, X):
        X = pd.DataFrame(X)
        one = pd.DataFrame(np.ones((X.shape[0], 1)))
        return pd.concat([X, one], axis=1)

    def sigmoid(self, z):  # sigmoid function
        return 1 / (1 + np.exp(-z))

    def gradient(self, X, y, theta):
        for iter_num in range(self.num_iter):
            grad = (-1 / y.shape[0]) * np.dot(X.T, self.sigmoid(np.dot(X, theta)) - y)
            theta = theta + grad * self.alpha
        return theta

    def predict(self, test, weight):
        array = []
        for item in test:
            array.append(list(item) + [1])
        test = np.asarray(array)
        logit = 1 / (1 + np.exp(-np.matmul(test, weight)))
        gender_list = np.where(logit > 0.5, 1, 0)
        return logit, gender_list

