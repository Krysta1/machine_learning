import numpy as np
import pandas as pd
from collections import Counter


class Logistic(object):
    def __init__(self):
        pass

    def x_plus_1(self, X):
        X = pd.DataFrame(X)
        one = pd.DataFrame(np.ones((X.shape[0], 1)))
        return pd.concat([X, one], axis=1)

    def load_data(self, path):  # load data
        data = np.loadtxt(path, delimiter=",", dtype=float)
        X = data[:, 0:-1]
        y = data[:, -1]
        return X, y

    def sigmoid(self, z):  # sigmoid function
        return 1 / (1 + np.exp(-z))

    def cost_func(self, X, y, theta):  # calculate the cost
        m, n = X.shape
        first = - np.log(self.sigmoid(np.dot(X, theta))) * y
        second = np.log(1 - self.sigmoid(np.dot(X, theta))) * (1 - y)
        return (1.0 / m) * (np.sum(first - second))

    def gradient(self, X, y, num_iter, alpha):
        X = self.x_plus_1(X)
        theta = np.zeros(X.shape[1])
        for iter_num in range(num_iter):
            cost = self.cost_func(X, y, theta)
            grad = (-1 / y.shape[0]) * np.dot(X.T, self.sigmoid(np.dot(X, theta)) - y)
            theta = theta + grad * alpha

            # if iter_num % 100 == 0:
                # print("iter: %d is cost: %12f" % (iter_num, cost))
        print("thetaL ", theta)
        return theta, cost

    def predict(self, test_X, test_y, weight):
        array = []
        for item in test_X:
            array.append(list(item) + [1])
        test_X = np.asarray(array)
        logits = 1 / (1 + np.exp(-np.matmul(test_X, weight)))
        gender_list = np.where(logits > 0.5, "1", "0")
        # print(gender_list)
        count = 0
        for i in range(len(array)):
            if int(gender_list[i]) == test_y[i]:
                count += 1
            array[i].remove(array[i][-1])
            print("Predicting %s and the result is %s." % (array[i], gender_list[i]))
        print("The accuracy is %f" % (count / len(test_y)))
        return gender_list

    def load_sampling_data(self, path):
        data = np.loadtxt(path, delimiter=",", dtype=float)
        tmp = []
        for j in range(len(data)):
            tmp.append(list(data[(np.random.choice(range(len(data))))]))
        print(tmp)
        X = np.array(tmp)[:, 0:-1]
        y = np.array(tmp)[:, -1]
        t = []
        for x in tmp:
            if x not in t:
                t.append(x)
        print(len(t))
        return X, y


if __name__ == "__main__":
    alpha = 0.01
    num_iter = 5000

    lg = Logistic()
    # X, y = lg.load_sampling_data("train.txt")

    bagging_result = []
    for i in range(50):
        print("%dth iteration" % (i + 1))
        X, y = lg.load_sampling_data("train.txt")
        print(len(X))

        weight, cost = lg.gradient(X, y, num_iter, alpha)

        test_X, test_y = lg.load_data("test.txt")
        gender_list = lg.predict(test_X, test_y, weight)
        bagging_result.append(list(gender_list))

    # print(bagging_result)
    bagging_result = np.array(bagging_result).T
    print(bagging_result)
    final_result = []
    for x in bagging_result:
        final_result.append(Counter(x).most_common(1)[0][0])

    # print(len(final_result))

    count = 0
    for i in range(len(test_y)):
        if int(final_result[i]) == test_y[i]:
            count += 1
    print("The accuracy is %f" % (1.0 * count / len(test_y)))
