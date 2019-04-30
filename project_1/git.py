import numpy as np
import pandas as pd


def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    n = len(theta)

    temp = np.matrix(np.zeros((n, num_iters)))  # 暂存每次迭代计算的theta，转化为矩阵形式

    J_history = np.zeros((num_iters, 1))  # 记录每次迭代计算的代价值

    for i in range(num_iters):  # 遍历迭代次数
        h = np.dot(X, theta)  # 计算内积，matrix可以直接乘
        temp[:, i] = theta - ((alpha / m) * (np.dot(np.transpose(X), h - y)))  # 梯度的计算
        theta = temp[:, i]
        J_history[i] = computerCost(X, y, theta)  # 调用计算代价函数
        print('.', end=' ')
    return theta, J_history


# 计算代价函数
def computerCost(X, y, theta):
    m = len(y)
    J = 0

    J = (np.transpose(X * theta - y)) * (X * theta - y) / (2 * m)  # 计算代价J
    return J


def linearRegression(X, y, alpha=0.01, num_iters=400):
    # print(u"加载数据...\n")

    # data = pd.read_csv("data.csv", ",", np.float64)  # 读取数据
    # X = data[:, 0:-1]  # X对应0到倒数第2列
    # y = data[:, -1]  # y对应最后一列
    m = len(y)  # 总的数据条数
    col = X.shape[1] + 1  # data的列数

    # X, mu, sigma = featureNormaliza(X)  # 归一化
    # plot_X1_X2(X)  # 画图看一下归一化效果

    X = np.hstack((np.ones((m, 1)), X))  # 在X前加一列1

    print(u"\n执行梯度下降算法....\n")

    theta = np.zeros((col, 1))
    y = y.reshape(-1, 1)  # 将行向量转化为列
    theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)

    # plotJ(J_history, num_iters)

    return theta  # 返回均值mu,标准差sigma,和学习的结果theta

if __name__ == '__main__':
    data = np.loadtxt("data.txt")
    label = pd.DataFrame(data[:, -1])
    training_data = pd.DataFrame(data[:, 0:-1])
    order = 4
    training_x = pd.DataFrame([])
    for i in range(order):
        tmp = training_data ** (i + 1)
        training_x = pd.concat([training_x, tmp], axis=1, ignore_index=True)
    x = training_x
    y = np.array(pd.DataFrame(data[:, -1]))
    learning_rate = 0.001
    init_weight = np.zeros((2 * order + 1, 1))
    num_iter = 500
    print(linearRegression(x, y, learning_rate, num_iter))