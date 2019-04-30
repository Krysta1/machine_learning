import numpy as np
M = 1
W = 0
D = [ ((170, 57, 32), W),
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

class LogisticRegression(object):
    def __init__(self, dims):
        """
        initial model
        :param dims: dimensions of parameter
        """
        self.dims = dims
        self.W = np.random.uniform(size=[dims+1])
        self.model = lambda x: 1/(1+np.exp(-np.matmul(x,self.W)))

    def fit(self, data, iteration, learning_rate):
        """
        fit data using logistic regression by gradient descent
        :param data: data
        :param iteration: training number
        :param learning_rate: learning rate
        :return:
        """
        array = []
        for item in data:
            array.append(list(item[0])+[1,item[1]])
        data = np.asarray(array)
        for i in range(iteration):
            print("iteration:{}\n".format(i))
            # sum((yn-tn)x)
            diataW = np.mean(np.tile(np.expand_dims(self.model(data[:,:-1])-data[:,-1],1),[1,4])*data[:,:-1],axis=0)-0.0001*self.W
            self.W-=learning_rate*diataW
            print(self.W)

    def predict(self, test):
        array = []
        for item in test:
            array.append(list(item)+[1])
        test = np.asarray(array)
        logits = self.model(test)
        # print(logits)
        print(np.where(logits>0.5,1,0))
        # accuracy = np.mean(np.equal(logits,np.where(test[:,-1]>0.5,1,0)))
        return logits

if __name__=="__main__":
    lr = LogisticRegression(3)
    lr.fit(D,5000,0.001)
    # You can set diferent input
    T = [(155, 40, 35),(170, 70, 32),(175, 70, 35),(180, 90, 20)]
    print(lr.predict(T))