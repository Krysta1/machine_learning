import numpy as np
import pandas as pd
import matplotlib as plt


def sigmoid(z):  # sigmoid function
    return 1/(1 + np.exp(-z))


def cost_func(X, y, theta):  # cal cost
    m, n = X.shape

    # J = 0

    # grad = np.zeros(theta.shape)
    # theta2 = theta[range(1, theta.shape[0]), :]

    j = (-1.0 / m) * (np.sum(np.log(sigmoid(np.dot(X, theta))) * y + (
            np.log(1 - sigmoid(np.dot(X, theta))) * (1 - y))))
    return j


def gradient(theta, X, y, num_iters, alpha):  # cal the derivative
    # grad = np.zeros(theta.shape)  # （1,3）
    # error = np.sum((X.T @ (sigmoid(X @ theta) - y))) / len(X)
    # for j in range(len(theta.ravel())):  # for each parmeter
    #     term = np.multiply(error, X[:, j])
    #     grad[0, j] = np.sum(term) / len(X)
    #
    # return grad
    # alpha = self.alpha
    # debug = self.debug
    # num_iters = self.num_iters
    m, n = data.shape
    # regularized = self.regularized

    # print 'inoming regularized', regularized
    # if (regularized == True):
    #     llambda = 1
    # else:
    #     llambda = 0

    for eachIteration in range(num_iters):
        cost = cost_func(X, y, theta)
        # if (debug):
        #     print('iteration: ', eachIteration)
        #     print('cost: ', cost)

        # compute gradient

        B = sigmoid(np.dot(X, theta) - y)

        A = (1 / m) * np.transpose(X)

        grad = np.dot(A, B)

        A = (sigmoid(np.dot(X, theta)) - y)
        B = X[:, 0].reshape((X.shape[0], 1))

        grad[0] = (1 / m) * np.sum(A * B)

        A = (sigmoid(np.dot(X, theta)) - y)
        B = (X[:, range(1, n)])

        for i in range(1, len(grad)):
            A = (sigmoid(np.dot(X, theta)) - y)
            B = (X[:, i].reshape((X[:, i].shape[0], 1)))
            grad[i] = (1 / m) * np.sum(A * B) + ((1 / m) * theta[i])

        init_theta = theta - (np.dot((alpha / m), grad))

    return init_theta, cost
    # return (X.T @ (sigmoid(X @ theta) - y)) / len(X)


def gradient_descent(X, y, theta, num_iter, alpha):  # find opt theta
    grad = np.zeros(theta.shape)
    cost = [cost_func(X, y, theta)]
    for i in range(num_iter):
        grad = gradient(X, y, theta)
        theta = theta - alpha * grad
        cost.append(cost(X, y, theta))
    return theta, cost


def runExpe(X, y, theta, num_iter, alpha):
    #import pdb; pdb.set_trace();
    theta, cost = gradient(X, y, theta, num_iter, alpha)
    # name = "Original" if (data[:,1]>2).sum() > 1 else "Scaled"
    # name += " data - learning rate: {} - ".format(alpha)
    # if batchSize==n: strDescType = "Gradient"
    # elif batchSize==1:  strDescType = "Stochastic"
    # else: strDescType = "Mini-batch ({})".format(batchSize)
    # name += strDescType + " descent - Stop: "
    # if stopType == STOP_ITER: strStop = "{} iterations".format(thresh)
    # elif stopType == STOP_COST: strStop = "costs change < {}".format(thresh)
    # else: strStop = "gradient norm < {}".format(thresh)
    # name += strStop
    # print ("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
    #     name, theta, iter, costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(cost)), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    # ax.set_title(name.upper() + ' - Error vs. Iteration')
    return theta


def predict(theta):  # predict the result
    pass


if __name__ == "__main__":
    pd.read_csv("gender_data.csv")

    data = pd.read_csv("gender_data.csv", dtype={'gender': str})
    class_mapping = {'W': 0, 'M': 1}
    data['gender'] = data['gender'].map(class_mapping)
    X = data.iloc[:, 0:-1].values
    y = data.iloc[:, -1].values

    alpha = 0.001
    num_iter = 5000
    theta = np.zeros(X.shape[1])
    print(X.shape, y.shape, theta.shape)

    runExpe(X, y, theta, num_iter, alpha)
