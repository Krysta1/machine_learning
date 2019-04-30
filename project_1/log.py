import numpy as np
import pandas as pd


def x_plus_1(X):
    X = pd.DataFrame(X)
    one = pd.DataFrame(np.ones((X.shape[0], 1)))
    return pd.concat([X, one], axis=1)


def load_data():  # load data
    data = pd.read_csv("gender_data.csv", dtype={'gender': str})
    class_mapping = {'W': 0, 'M': 1}
    data['gender'] = data['gender'].map(class_mapping)
    X = data.iloc[:, 0:-1].values
    y = data.iloc[:, -1].values
    return X, y


def sigmoid(z):  # sigmoid function
    return 1/(1 + np.exp(-z))


def cost_func(X, y, theta):  # calculate the cost
    m, n = X.shape
    first = - np.log(sigmoid(np.dot(X, theta))) * y
    second = np.log(1 - sigmoid(np.dot(X, theta))) * (1 - y)
    return (1.0 / m) * (np.sum(first - second))


def gradient(X, y, theta, num_iter, alpha):
    m, n = X.shape

    for iter_num in range(num_iter):
        cost = cost_func(X, y, theta)
        grad = (-1 / y.shape[0]) * np.dot(X.T, sigmoid(np.dot(X, theta)) - y)
        theta = theta + grad * alpha

        if iter_num % 100 == 0:
            print("iter: %d is cost: %12f" % (iter_num, cost))

    return theta, cost


def predict(test, weight):
    array = []
    for item in test:
        array.append(list(item)+[1])
    # print(array)
    test = np.asarray(array)
    logits = 1/(1+np.exp(-np.matmul(test, weight)))
    # print(logits)
    gender_list = np.where(logits > 0.5, "M", "W")
    for i in range(len(array)):
        print("Predicting %s and the result is %s." % (array[i], gender_list[i]))
    return logits


if __name__ == "__main__":
    X, y = load_data()

    alpha = 0.001
    num_iter = 5000
    theta = np.zeros(X.shape[1] + 1)
    X = x_plus_1(X)
    weight, cost = gradient(X, y, theta, num_iter, alpha)
    T = [(155, 40, 35), (170, 70, 32), (175, 70, 35), (180, 90, 20)]
    predict(T, weight)



