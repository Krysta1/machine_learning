import numpy as np


def transfer(X, y):  # add ones col to X, and transfer 0 in y to -1
    x0 = np.ones((X.shape[0], 1))
    X = np.c_[X, x0]
    y[y == 0] = -1
    return X, y


class Logistic(object):

    def __init__(self, learning_rate=0.1, num_iter=100):
        self.theta = None  # init the theta of logistic
        self.learning_rate = learning_rate  # init the learning rate
        self.num_iter = num_iter  # init the iteration times

    def load_data(self, path):  # load data
        data = np.loadtxt(path, delimiter=",", dtype=float)
        X = data[:, 0:-1]
        y = data[:, -1]
        return X, y

    def sigmoid(self, z):  # sigmoid function
        return 1 / (1 + np.exp(-z))

    def gradient(self, X, y):  # Logistic gradient descent.
        self.theta = np.ones((X.shape[1], 1))
        for i in range(self.num_iter):
            for j in range(X.shape[0]):
                py = (y[j] / 2) + 0.5  # transfer y from -1 and 1 to 0 and 1 to get a better result
                h = self.sigmoid(np.dot(self.theta.T, X[j]))
                self.theta = self.theta + self.learning_rate * (py - h) * X[j].reshape((X.shape[1], 1))
        print("theta:", self.theta)
        return self.theta

    # Using theta to predict a new sample. And cal the error rate
    def predict(self, test_X, test_y, theta, output=False):
        logits = 1 / (1 + np.exp(-np.matmul(test_X, theta)))
        gender_list = np.where(logits > 0.5, 1, -1)
        count = 0
        for i in range(len(gender_list)):
            if int(gender_list[i]) != int(test_y[i]):
                count += 1
        if output:
            print("Wrong predtion number: %d" % count)
            print("Error rate: %f" % (count / len(test_y)))
        return gender_list

    def sampled_data(self, path):  # generate random sampled data
        data = np.loadtxt(path, delimiter=",", dtype=float)
        tmp = []
        for j in range(len(data)):
            tmp.append(list(data[(np.random.choice(range(len(data))))]))
        X = np.array(tmp)[:, 0:-1]
        y = np.array(tmp)[:, -1]
        t = []
        for x in tmp:
            if x not in t:
                t.append(x)
        print("This bag totally has %d different samples." % len(t))
        return X, y
