import numpy as np
import pandas as pd

def liner_Regression(data_x,data_y,learningRate,Loopnum):
    Weight=np.ones(shape=(1,data_x.shape[1]))
    baise=np.array([[1]])

    for num in range(Loopnum):
        WXPlusB = np.dot(data_x, Weight.T) + baise

        loss=np.dot((data_y-WXPlusB).T,data_y-WXPlusB)/data_y.shape[0]
        w_gradient = -(2/data_x.shape[0])*np.dot((data_y-WXPlusB).T,data_x)
        baise_gradient = -2*np.dot((data_y-WXPlusB).T,np.ones(shape=[data_x.shape[0],1]))/data_x.shape[0]

        Weight=Weight-learningRate*w_gradient
        baise=baise-learningRate*baise_gradient
        # print("loss##########")
        # if num%50==0:
        #     print(loss)

    print(Weight)
    print(baise)
    return (Weight,baise)



if __name__== "__main__":
    # Weights = np.array([[3, 4, 6, 7]])
    # 处理数据
    df = pd.read_csv('data1.txt', sep=',', header=None)
    # print(df)
    data = np.array(df)
    data_x = np.matrix(data[:, :4])
    data_y = data[:, -1]
    for i in range(len(data_y)):
        if data_y[i] == 'Iris-setosa':
            data_y[i] = 1
        elif data_y[i] == 'Iris-versicolor':
            data_y[i] = 2
        else:
            data_y[i] = 3

    data_y = data_y.reshape((150, 1))
    print("data_x######")
    print(data_x)
    print("data_y######")
    print(data_y.shape)
    print("###################")

    res=liner_Regression(data_x, data_y, learningRate=0.003, Loopnum=1000)
    print("weight########")
    print(res[0].shape)
    print("baise########")
    print(res[1])
