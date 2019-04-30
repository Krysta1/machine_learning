import numpy as np
from matplotlib import pyplot as plt

# TD is the training data, woman = 0; man = 1
W = 0
M = 1
TD = [((170, 57, 32), W),
      ((192, 95, 28), M),
      ((150, 45, 30), W),
      ((170, 65, 29), M),
      ((175, 78, 35), M),
      ((185, 90, 32), M),
      ((170, 65, 28), W),
      ((155, 48, 31), W),
      ((160, 55, 30), W),
      ((182, 80, 30), M),
      ((175, 69, 28), W),
      ((180, 80, 27), M),
      ((160, 50, 31), W),
      ((175, 72, 30), M)]


class LDA(object):
    def __init__(self):
        pass

    def fit(self, data):
        """
        using Training data to calculation the Various parameters
        :param data: Training data
        :return:
        """
        Wdata = []
        Mdata = []
        for k in data:
            if k[1] == W:
                Wdata.append(list(k[0]))
            else:
                Mdata.append(k[0])
        # Put data with attributes for woman in wdata
        wdata = np.asarray(Wdata)
        print(wdata)

        # Put data with attributes for men in mdata
        mdata = np.asarray(Mdata)
        print(mdata)

        # calculation the mean of woman and man
        self.wmean = np.expand_dims(np.mean(wdata, 0), 1)
        print(self.wmean)
        self.mmean = np.expand_dims(np.mean(mdata, 0), 1)

        # calculation the Covariance of man and woman
        print((wdata.T - self.wmean).shape, ((wdata.T - self.wmean).T).shape)
        Covariance1 = np.matmul(wdata.T - self.wmean, (wdata.T - self.wmean).T)
        Covariance2 = np.matmul(mdata.T - self.mmean, (mdata.T - self.mmean).T)

        # calculation the Covariance of Training data
        self.Covariance = (Covariance1 + Covariance2) / (len(data) - 2)

        self.W = np.dot(np.linalg.inv(self.Covariance), self.wmean - self.mmean)
        print(self.W)
        self.b = -1 / 2 * np.dot((self.wmean + self.mmean).T, self.W) + np.log(wdata.shape[0] / mdata.shape[0])
        return wdata, mdata

    def plot(self, wdata, mdata):
        '''
        This function is used to Drawing
        :return:
        '''
        # print("nmb", np.reshape(self.wmean, [3]))
        wdata1 = np.random.multivariate_normal(np.reshape(self.wmean, [3]), self.Covariance, 50)
        mdata2 = np.random.multivariate_normal(np.reshape(self.mmean, [3]), self.Covariance, 50)
        print("nmb", wdata1)
        plt.figure()
        x = np.arange(100, 240, 1)
        y = (-x * float(self.W[0]) - float(self.b)) / float(self.W[1])
        plt.plot(x, y, 'r')
        # Drawing the point of generated data.red is woman; blue is man.
        plt.scatter(wdata1[:, 0], wdata1[:, 1], c='r')
        plt.scatter(mdata2[:, 0], mdata2[:, 1], c='b')

        # Drawing the point of training data.yellow is woman; black is man.
        plt.scatter(wdata[:, 0], wdata[:, 1], c='y')
        plt.scatter(mdata[:, 0], mdata[:, 1], c='black')
        plt.show()

    def predict(self, testData):
        '''
        This function is used to predict the testData
        :param testData:
        :return: 0 / 1 that mean Man or Woman
        '''
        testData = np.asarray(testData)
        test_Value = np.matmul(testData, self.W) + self.b
        print(test_Value)
        if test_Value > 0:
            return W
        else:
            return M


if __name__ == "__main__":
    lda = LDA()
    wdata, mdata = lda.fit(TD)
    lda.plot(wdata, mdata)
    res1 = lda.predict([(155, 40, 35)])
    if res1 == 0:
        print('The predict value is :Woman')
    else:
        print('The predict value is :Man')
    print('-----------------------------------')
    res2 = lda.predict([(170, 70, 32)])
    if res2 == 0:
        print('The predict value is :Woman')
    else:
        print('The predict value is :Man')
    print('-----------------------------------')
    res3 = lda.predict([(175, 70, 35)])
    if res3 == 0:
        print('The predict value is :Woman')
    else:
        print('The predict value is :Man')
    print('-----------------------------------')
    res4 = lda.predict([(180, 90, 20)])
    if res4 == 0:
        print('The predict value is :Woman')
    else:
        print('The predict value is :Man')
    print('-----------------------------------')
