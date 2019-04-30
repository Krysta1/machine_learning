import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_training = pd.read_csv("MushroomTrain.csv", header=None)
data_test = pd.read_csv("MushroomTest.csv", header=None)

data_training = np.array(data_training)


def cal_entropy(prob_e, prob_p):
    if prob_e == 0:
        return -(prob_p * np.log2(prob_p))
    elif prob_p == 0:
        return -(prob_e * np.log2(prob_e))
    else:
        return -(prob_e * np.log2(prob_e) + prob_p * np.log2(prob_p))


def entropy(data_training, axis):
    if axis == 1:
        dict_e = {'b': 0, 'c': 0, 'x': 0, 'f': 0, 'k': 0, 's': 0}
        dict_p = {'b': 0, 'c': 0, 'x': 0, 'f': 0, 'k': 0, 's': 0}
    elif axis == 2:
        dict_e = {'f': 0, 'g': 0, 'y': 0, 's': 0}
        dict_p = {'f': 0, 'g': 0, 'y': 0, 's': 0}
    elif axis == 3:
        dict_e = {'n': 0, 'b': 0, 'c': 0, 'g': 0, 'r': 0, 'p': 0, 'u': 0, 'e': 0, 'w': 0, 'y': 0, 't': 0, 'f': 0}
        dict_p = {'n': 0, 'b': 0, 'c': 0, 'g': 0, 'r': 0, 'p': 0, 'u': 0, 'e': 0, 'w': 0, 'y': 0, 't': 0, 'f': 0}
    elif axis == 4:
        dict_e = {'t': 0, 'f': 0}
        dict_p = {'t': 0, 'f': 0}
    else:
        dict_e = {'a': 0, 'l': 0, 'c': 0, 'y': 0, 'f': 0, 'm': 0, 'n': 0, 'p': 0, 's': 0}
        dict_p = {'a': 0, 'l': 0, 'c': 0, 'y': 0, 'f': 0, 'm': 0, 'n': 0, 'p': 0, 's': 0}

    count = 0
    for data in data_training:
        if data[0] == 'e':
            dict_e[data[axis]] += 1
        else:
            dict_p[data[axis]] += 1

    tmp = 0
    for key in dict_e:
        if dict_e[key] + dict_p[key] != 0:
            prob_e = dict_e[key] / (dict_e[key] + dict_p[key])
            prob_p = dict_p[key] / (dict_e[key] + dict_p[key])
            tmp += cal_entropy(prob_e, prob_p) * ((dict_e[key] + dict_p[key]) / len(data_training))
    return tmp, dict_e, dict_p


l = []
features = ['shape', 'surface', 'color', 'bruises', 'odor']
for i in range(5):
    l.append(entropy(data_training, i + 1)[0])
print(l)
print("The %s feature has the minimum entropy." % features[l.index(min(l))])

print(entropy(data_training, 5))
