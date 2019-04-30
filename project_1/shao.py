import numpy as np

from matplotlib import pyplot as plt
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

T = [((0.8552, 1.8292), 11.5848), ((2.6248, 2.3993), 17.6138),
     ((8.0101, 8.8651), 54.1331), ((0.2922, 0.2867), 5.7326),
     ((9.2885, 4.8990), 46.3750), ((7.3033, 1.6793), 29.4356),
     ((4.8861, 9.7868), 46.4227), ((5.7853, 7.1269), 40.7433),
     ((2.3728, 5.0047), 24.6220), ((4.5885, 4.7109), 29.7602)]


class Polynomial(object):
    def __init__(self, order):
        """
        initial binary polynomial model
        :param order: order of polynomial
        """
        self.order = order
        # the new features. For example, new features of 1 order polynomial are [x1,y1], 2 order are [x1,x2,x1*x2], etc.
        self.new_features = []
        for i in range(1, order + 1):  # x1
            for j in range(i + 1):  # x2
                self.new_features.append((j, i - j))
        print(self.new_features)

    def _transform_features(self, item, features_map):
        """
        transform a pice of origin data to new features data
        :param item: a piece of data
        :param features_map: self.new_features
        :return:
        """
        new_input = [1]
        for j in range(len(features_map)):
            new_input.append((item[0][0] ** features_map[j][0]) * (item[0][1] ** features_map[j][1]))
            # print(new_input)
        # print(new_input)
        return new_input

    def fit(self, data):
        """
        fit data using Polynomial model
        :param data: the dataset
        :return:
        """
        self.data = []
        self.new_data = []
        for i in range(len(data)):
            self.new_data.append(self._transform_features(data[i], self.new_features))
            self.data.append([data[i][0][0], data[i][0][1], data[i][1]])
        print(self.data)
        self.data = np.asarray(self.data)  # 将列表转化为矩阵
        self.W = np.matmul(np.linalg.pinv(self.new_data), self.data[:, -1])
        self.plot(D)

    def plot(self, data):
        # plot the polynomial surface and data point
        array = []
        for i in range(len(data)):
            array.append([data[i][0][0], data[i][0][1], data[i][1]])
        array = np.asarray(array)
        fig = plt.figure()
        ax = Axes3D(fig)
        x = np.arange(0, 10, 0.1)
        y = np.arange(0, 10, 0.1)
        z = np.arange(0, 100, 1)
        X, Y = np.meshgrid(x, y)  # 网格的创建，这个是关键
        Z = self.W[0]
        for i, f in enumerate(self.new_features):
            Z += self.W[i + 1] * (X ** f[0]) * (Y ** f[1])
        plt.xlabel('x')
        plt.ylabel('y')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
        ax.scatter(array[:, 0], array[:, 1], array[:, -1], c='b')
        plt.show()

    def predict(self, item):
        new_item = self._transform_features(item, self.new_features)
        logits = np.matmul(self.W, new_item)
        MSE = (logits - item[1]) ** 2
        # self.plot([item,[item[0],logits]])
        return logits, MSE


if __name__ == "__main__":
    poly = Polynomial(4)
    poly.fit(D)
    print(poly.W)

    for item in T:
        print(item, poly.predict(item))
