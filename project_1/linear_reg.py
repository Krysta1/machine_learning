import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

D = [((6.4432, 9.6309), 50.9155), ((3.7861, 5.4681), 29.9852),
     ((8.1158, 5.2114), 42.9626), ((5.3283, 2.3159), 24.7445),
     ((3.5073, 4.8890), 27.3704), ((9.3900, 6.2406), 51.1350),
     ((8.7594, 6.7914), 50.5774), ((5.5016, 3.9552), 30.5206),
     ((6.2248, 3.6744), 31.7380), ((5.8704, 9.8798), 49.6374),
     ((2.0774, 0.3774), 10.0634), ((3.0125, 8.8517), 38.0517),
     ((4.7092, 9.1329), 43.5320), ((2.3049, 7.9618), 33.2198),
     ((8.4431, 0.9871), 31.1220), ((1.9476, 2.6187), 16.2934),
     ((2.2592, 3.3536), 19.3899), ((1.7071, 6.7973), 28.4807),
     ((2.2766, 1.3655), 13.6945), ((4.3570, 7.2123), 36.9220),
     ((3.1110, 1.0676), 14.9160), ((9.2338, 6.5376), 51.2371),
     ((4.3021, 4.9417), 29.8112), ((1.8482, 7.7905), 32.0336),
     ((9.0488, 7.1504), 52.5188), ((9.7975, 9.0372), 61.6658),
     ((4.3887, 8.9092), 42.2733), ((1.1112, 3.3416), 16.5052),
     ((2.5806, 6.9875), 31.3369), ((4.0872, 1.9781), 19.9475),
     ((5.9490, 0.3054), 20.4239), ((2.6221, 7.4407), 32.6062),
     ((6.0284, 5.0002), 35.1676), ((7.1122, 4.7992), 38.2211),
     ((2.2175, 9.0472), 36.4109), ((1.1742, 6.0987), 25.0108),
     ((2.9668, 6.1767), 29.8861), ((3.1878, 8.5944), 37.9213),
     ((4.2417, 8.0549), 38.8327), ((5.0786, 5.7672), 34.4707)]


# creat the data by the order given
def creat_data_by_order(item, order):
    new_features = []
    for i in range(1, order + 1):  # x1
        for j in range(i + 1):  # x2
            new_features.append((j, i - j))

    new_data = []
    for i in range(len(item)):
        new_input = [1]

        for j in range(len(new_features)):
            new_input.append((item[i][0] ** new_features[j][0]) * (item[i][1] ** new_features[j][1]))

        new_data.append(new_input)

    new_data = np.array(new_data)
    # print(new_data.shape)
    return new_data, new_features


# gradient descent to get the best weights
def linear_regression(data_x, data_y, learningRate, Loopnum):
    print(data_x.shape, data_y.shape)
    weight = np.ones(shape=(1, data_x.shape[1])).T
    for num in range(Loopnum):
        k = np.dot(np.transpose(data_x), (np.dot(data_x, weight) - data_y))
        # print(k.shape)
        weight -= learningRate * k / len(data_y)
    print("The weights is: \n", weight)
    return weight


# predict the test data and calculate the total square error
def predict(test_x, test_y, weight):
    m, n = test_x.shape
    total_error = 0
    for i in range(m):
        prediction = np.dot(test_x[i, :], weight)
        error = test_y[i] - prediction
        print("Predicting: %s......" % test_x[0, :])
        print("Result: %f .True value: %f.Error: %f." % (prediction, test_y[i], error))
        print("-------------------------------------")
        total_error = error ** 2
    return total_error


# set learning rate and iteration numbers to get a better result.
def set_parameters(order):
    if order == 1 or order == 2:
        learning_rate = 0.0001
        num_iter = 50000
    elif order == 3:
        learning_rate = 0.0000001
        num_iter = 500000
    else:
        learning_rate = 0.00000001
        num_iter = 5000000
    return learning_rate, num_iter


# load training data
def load_training_data():
    data = np.loadtxt("data.txt")
    label = pd.DataFrame(data[:, -1])
    training_data = pd.DataFrame(data[:, 0:-1])

    x = training_data
    x = np.array(x)
    training_data, new_features = creat_data_by_order(x, order)
    y = np.array(pd.DataFrame(data[:, -1]))

    return training_data, y, new_features


# load test data
def load_test_data():
    test_data = np.loadtxt('test_data')
    test_data_x = pd.DataFrame(test_data[:, 0:-1])
    test_data_y = pd.DataFrame(test_data[:, -1])
    test_x = np.array(test_data_x)
    test_x, new_features = creat_data_by_order(test_x, order)
    test_y = np.array(pd.DataFrame(test_data[:, -1]))

    return test_x, test_y


# plot the 3D picture
def draw(data, weight, new_features):
    weight = [x for i in weight for x in i]
    array = []
    for i in range(len(data)):
        array.append([data[i][0][0], data[i][0][1], data[i][1]])
    array = np.asarray(array)
    fig = plt.figure()
    ax = Axes3D(fig)
    x = np.arange(0, 10, 0.1)
    y = np.arange(0, 10, 0.1)
    z = np.arange(0, 100, 1)
    X, Y = np.meshgrid(x, y)
    Z = weight[0]
    for i, f in enumerate(new_features):
        Z += weight[i + 1] * (X ** f[0]) * (Y ** f[1])
    plt.xlabel('x')
    plt.ylabel('y')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    ax.scatter(array[:, 0], array[:, 1], array[:, -1], c='b')
    plt.show()


if __name__ == '__main__':
    order = 1  # change the order value to handle different cases
    learning_rate, num_iter = set_parameters(order)

    training_data, y, new_features = load_training_data()
    weight = linear_regression(training_data, y, learning_rate, num_iter)

    test_x, test_y = load_test_data()
    print("Total square error of order %d is %12f" % (order, predict(test_x, test_y, weight)))

    # draw(D, weight, new_features)
