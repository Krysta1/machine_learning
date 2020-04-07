import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data():  # load the training data and test data
    train = pd.read_csv('training.txt', sep=',', header=None, dtype={3: str})
    test = pd.read_csv('test.txt', sep=',', header=None, dtype={3: str})

    train[3] = train[3].replace(['M', 'W'], [1, 0])
    test[3] = test[3].replace(['M', 'W'], [1, 0])

    train_man = train.loc[(train[3] == 1)].iloc[:, 0:-1]
    train_woman = train.loc[(train[3] == 0)].iloc[:, 0:-1]
    #     print(train_man)
    train_man = np.mat(train_man).T
    train_woman = np.mat(train_woman).T

    test_man = test.loc[(test[3] == 1)].iloc[:, 0:-1]
    test_woman = test.loc[(test[3] == 0)].iloc[:, 0:-1]
    test_man = np.mat(test_man)
    test_woman = np.mat(test_woman)

    return train_man, train_woman, test_man, test_woman


def generate_data(man, woman):  # generate the data of man and women
    man_T, woman_T = man.T, woman.T
    w_mean = np.mean(woman_T, 0).T
    m_mean = np.mean(man_T, 0).T
    cov1 = np.matmul(man_T.T - m_mean, (man_T.T - m_mean).T)
    cov2 = np.matmul(woman_T.T - w_mean, (woman_T.T - w_mean).T)
    cov = (cov1 + cov2) / (len(man_T) + len(woman_T) - 2)
    mean_women = np.reshape(w_mean, [3]).tolist()[0]
    mean_man = np.reshape(m_mean, [3]).tolist()[0]
    generate_w_data = np.random.multivariate_normal(mean_women, cov, 50)
    generate_m_data = np.random.multivariate_normal(mean_man, cov, 50)
    return generate_m_data, generate_w_data


def draw(man, woman, w, mean1, mean2):  # plot a picture of all the original data and generated data.
    # x = np.arange(130, 200, 5)
    # tmp = w[0][0] / w[1][0]
    # # calculate the y values for given x
    # y = - tmp * x + 0.5 * tmp * (mean1[0][0] + mean2[0][0]) + 0.5 * (mean1[1][0] + mean2[1][0])
    # y = np.array(y).T

    generate_m_data, generate_w_data = generate_data(man, woman)
    plt.ylim(20, 110)
    plt.xlim(130, 200)
    # plt.plot(x, y, 'r')
    plt.xlabel("Height")
    plt.ylabel("Weight")
    # draw generated data: man in blue and woman in black
    plt.scatter(generate_w_data[:, 0], generate_w_data[:, 1], c='r')
    plt.scatter(generate_m_data[:, 0], generate_m_data[:, 1], c='b')
    # draw original data: man in yellow and woman in green
    plt.scatter(man[0, :].tolist(), man[1, :].tolist(), c='black')
    plt.scatter(woman[0, :].tolist(), woman[1, :].tolist(), c='g')
    plt.show()


def compute_mean(samples):  # calculate the mean
    mean_mat = np.mean(samples, axis=1)
    return mean_mat


def cal_inclass_scatter(samples, mean):  # calculate the in class scatter
    dimens, nums = samples.shape[:2]
    samples_mean = samples - mean
    s_in = 0
    for i in range(nums):
        x = samples_mean[:, i]
        s_in += np.dot(x, x.T)
    return s_in


def precdict(test_list, w, mean1, mean2):  # predict the test data in test list
    for test in test_list:
        test = np.array(test)
        if np.dot(w.T, test.T - 0.5 * (mean1 + mean2)) > 0:
            res = "Man"
        else:
            res = "Woman"
        print("Predicting %s and the result is %s" % (test[0], res))
    return res


if __name__ == '__main__':
    training_man, training_women, test_man, test_woman = load_data()

    # calculate the mean of two classes
    mean1 = compute_mean(training_man)
    mean2 = compute_mean(training_women)

    # calculate the with in class scatter of two classes
    scatter_in1 = cal_inclass_scatter(training_man, mean1)
    scatter_in2 = cal_inclass_scatter(training_women, mean2)
    scatter_in = (scatter_in1 + scatter_in2)

    # calculate the weights
    w = np.dot(scatter_in.I, mean1 - mean2)

    # plot a picture with original data and generated data and decision boundary
    draw(training_man, training_women, w, mean1, mean2)

    # predict the woman test set and man test set
    precdict(test_woman, w, mean1, mean2)
    precdict(test_man, w, mean1, mean2)
