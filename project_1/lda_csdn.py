import numpy as np
import pandas as pd
# from numpy import *
import matplotlib.pyplot as plt


def load_data():
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

    #     print(train_man, train_woman)

    return train_man, train_woman, test_man, test_woman


def draw(group, w, mean1, mean2):
    x = np.arange(130, 200, 5)
    tmp = w[0][0] / w[1][0]
    y = - tmp * x + 0.5 * tmp * (mean1[0][0] + mean2[0][0]) + 0.5 * (mean1[1][0] + mean2[1][0])
    y = np.array(y).T
    print(y.shape)

    print(group.shape)
    fig = plt.figure()
    plt.ylim(20, 150)
    plt.xlim(130, 200)
    plt.plot(x, y, 'b')
    ax = fig.add_subplot(111)
    #     ax.scatter(group[0,:].tolist(), group[1,:].tolist())
    ax.scatter(group[0, 0:7].tolist(), group[1, 0:7].tolist(), c='r')
    ax.scatter(group[0, 7:].tolist(), group[1, 7:].tolist(), c='g')
    plt.show()
    # end of draw


# 计算样本均值
# 参数samples为nxm维矩阵，其中n表示维数，m表示样本个数
def compute_mean(samples):
    mean_mat = np.mean(samples, axis=1)
    return mean_mat
    # end of compute_mean


# 计算样本类内离散度
# 参数samples表示样本向量矩阵，大小为nxm，其中n表示维数，m表示样本个数
# 参数mean表示均值向量，大小为1xd，d表示维数，大小与样本维数相同，即d=m
def compute_withinclass_scatter(samples, mean):
    # 获取样本维数，样本个数
    print(samples.shape)
    dimens, nums = samples.shape[:2]
    # 将所有样本向量减去均值向量
    samples_mean = samples - mean
    # 初始化类内离散度矩阵
    s_in = 0
    for i in range(nums):
        x = samples_mean[:, i]
        s_in += np.dot(x, x.T)
    # endfor
    return s_in
    # end of compute_mean


def precdict(test_list, w, mean1, mean2):
    #     test_list = [x.tolist() for i in test_list for x in i]
    for test in test_list:
        test = np.array(test)
        print(test)
        #         print(test.shape)
        #         print((mean1 - mean2).shape)
        print(np.dot(w.T, test.T - 0.5 * (mean1 + mean2)))
        if np.dot(w.T, test.T - 0.5 * (mean1 + mean2)) > 0:
            a = 1
        else:
            a = 0
        #         a = if (np.dot(w.T, test.T - 0.5 * (mean1 - mean2)) > 0) 1 else 0
        print(a)
    return a


if __name__ == '__main__':
    # group1, group2 = createDataSet()
    group1, group2, test_man, test_woman = load_data()
    print(test_man, test_woman)
    print("group1 :\n", group1)
    print("group2 :\n", group2)

    mean1 = compute_mean(group1)
    print("mean1 :\n", mean1)
    mean2 = compute_mean(group2)
    print("mean2 :\n", mean2)
    s_in1 = compute_withinclass_scatter(group1, mean1)
    print("s_in1 :\n", s_in1)
    s_in2 = compute_withinclass_scatter(group2, mean2)
    print("s_in2 :\n", s_in2)
    # 求总类内离散度矩阵
    s = s_in1 + s_in2
    print("s :\n", s)
    # 求s的逆矩阵
    #     s_t = np.linalg.inv(s)
    s_t = s.I
    print("s_t :\n", s_t)
    # 求解权向量
    w = np.dot(s_t, mean1 - mean2)
    draw(np.hstack((group1, group2)), w, mean1, mean2)
    print("w :\n", w)
    precdict(test_woman, w, mean1, mean2)
