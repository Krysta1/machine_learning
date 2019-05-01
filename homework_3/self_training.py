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


def cal_accuracy(list1, list2):
    count = 0
    for i in range(len(list1)):
        if list1[i] == list2[i]:
            count += 1
    return 1.0 * count / len(list1)


d_s = [[170, 57, 32, 0],
       [190, 95, 28, 1],
       [150, 45, 35, 0],
       [168, 65, 29, 1],
       [175, 78, 26, 1],
       [185, 90, 32, 1],
       [171, 65, 28, 0],
       [155, 48, 31, 0],
       [165, 60, 27, 0]]

d_t = [[169, 58, 30, 0],
       [185, 90, 29, 1],
       [148, 40, 31, 0],
       [177, 80, 29, 1],
       [170, 62, 27, 0],
       [172, 72, 30, 1],
       [175, 68, 27, 0],
       [178, 80, 29, 1]]

d_u = [[182, 80, 30], [175, 69, 28], [178, 80, 27], [160, 50, 31], [170, 72, 30], [152, 45, 29], [177, 79, 28],
       [171, 62, 27], [185, 90, 30], [181, 83, 28], [168, 59, 24], [158, 45, 28], [178, 82, 28], [165, 55, 30],
       [162, 58, 28], [180, 80, 29], [173, 75, 28], [172, 65, 27], [160, 51, 29], [178, 77, 28], [182, 84, 27],
       [175, 67, 28], [163, 50, 27], [177, 80, 30], [170, 65, 28]]


def supervised(test_X, test_y):
    training = np.array(d_s)
    normal_lr = LogisticReg()
    X, y = training[:, :-1], training[:, -1]
    theta = np.zeros(X.shape[1] + 1)
    X = normal_lr.x_plus_1(X)
    weight = normal_lr.gradient(X, y, theta)
    nor_prediction, nor_gender = normal_lr.predict(test_X, weight)
    # print(nor_prediction)
    print("Predict results of normal supervised learning: {}".format(nor_gender))
    print("Test set labels: {}".format(test_y))
    print("The accuracy of normal classifier is {}".format(cal_accuracy(nor_gender, test_y)))


def semi_supervised(test_X, test_y):
    lr = LogisticReg()
    while True:
        training = np.array(d_s)
        X, y = training[:, :-1], training[:, -1]
        theta = np.zeros(X.shape[1] + 1)
        X = lr.x_plus_1(X)
        weight = lr.gradient(X, y, theta)
        prediction_res, gender = lr.predict(d_u, weight)[0].tolist(), lr.predict(d_u, weight)[1]

        if min(prediction_res) - 0 > 1 - max(prediction_res):
            tmp = max(prediction_res)
        else:
            tmp = min(prediction_res)

        index = prediction_res.index(tmp)
        label = 1 if prediction_res[index] > 0.5 else 0
        d_s.append([d_u[index][0], d_u[index][1], d_u[index][2], label])
        d_u.remove(d_u[index])

        if len(d_u) == 0:
            break
    prediction, gender_list = lr.predict(test_X, weight)
    print("Predict results of semi-supervised learning: {}".format(gender_list))
    print("Test set labels: {}".format(test_y))
    print("The accuracy of semi-supervised classifier is {}".format(cal_accuracy(gender_list, test_y)))


if __name__ == "__main__":
    training = np.array(d_t)
    test_X, test_y = training[:, :-1], training[:, -1]
    supervised(test_X, test_y)
    print("---------------------------")
    semi_supervised(test_X, test_y)
