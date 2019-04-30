from project_2.logistic_reg import *
import numpy as np

boost_times = 50
alpha_dic = []
theta_dic = {}


def cal_W(W, alpha, y, pred):
    updated_W = []
    for i in range(len(y)):
        updated_W.append(W[i] * np.exp(-alpha * y[i] * pred[i]))
    return np.array(updated_W / sum(updated_W)).reshape([len(y), 1])


def cal_e(y, pred, W):
    tmp = 0
    for i in range(len(y)):
        if int(y[i]) != int(pred[i]):
            tmp += W[i]
    return tmp


def cal_alpha(e):
    if e == 0:
        return 10000
    elif e == 0.5:
        return 0.001
    else:
        return 0.5 * np.log((1 - e) / e)


def train_boost(X, y, times):
    print("Start %d times boosting: " % boost_times)
    boost_weight = np.ones(len(X)) * (1 / len(X))
    theta_dic.setdefault(times-1)

    for i in range(times):
        print("------------------------------------")
        print("%d th iteration" % (i + 1))
        dig_weight = np.diag(boost_weight)
        X_with_weight = np.dot(dig_weight, X)
        # print("Weighted data: %s" % X_with_weight)
        theta = lg.gradient(X_with_weight, y)
        gender_list = lg.predict(X, y, theta, True)
        e = cal_e(y, gender_list, boost_weight)

        print("e in this iteration: %f" % e)
        al = cal_alpha(e)
        alpha_dic.append(al)
        theta_dic[i] = theta

        boost_weight = cal_W(boost_weight, al, y, gender_list)
        boost_weight = boost_weight.flatten().tolist()
    # print(alpha_dic)
    # print(theta_dic)


def boost_predict(X, y, model_weight, model):
    ret_list = []
    for i in range(len(model)):
        tmp = lg.predict(X, y, model[i])
        tmp = tmp.reshape((1, tmp.shape[0])).tolist()
        ret_list.append(tmp[0])

    model_weight_diag = np.diag(model_weight)
    weighted_result = np.dot(model_weight_diag, np.array(ret_list))

    ret = np.zeros((1, X.shape[0]))
    for result in weighted_result:
        ret += result
    final_ret = np.sign(ret)

    cnt = 0
    for i in range(len(y)):
        if final_ret[0][i] != y[i]:
            cnt += 1
    print("------------------------------------")
    print("After boosting, final result:")
    print("After boosting, final error number is %d" % cnt)
    print("After boosting, final error rate is %f" % (cnt / len(y)))


if __name__ == "__main__":
    lg = Logistic(learning_rate=0.1, num_iter=100)

    X, y = lg.load_data("train.txt")
    X, y = transfer(X, y)

    test_X, test_y = lg.load_data("test.txt")
    test_X, test_y = transfer(test_X, test_y)

    train_boost(X, y, boost_times)
    # boost_predict(np.array(X), y, alpha_dic, theta_dic)
    boost_predict(np.array(test_X), test_y, alpha_dic, theta_dic)
