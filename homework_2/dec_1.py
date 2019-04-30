# import numpy as np
# import pandas as pd
from math import log
import operator


def ShannonEnt(dataSet):
    num = len(dataSet)
    labelCount = {}
    for data in dataSet:
        curLabel = data[-1]
        if curLabel not in labelCount.keys():
            labelCount[curLabel] = 0
        labelCount[curLabel] += 1
    shannonEnt = 0.0
    for key in labelCount:
        prob = float(labelCount[key]) / num
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataset(dataSet, axis, value):
    newDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            newDataSet.append(featVec)
    return newDataSet


def creatDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    label = ['no surfacing', 'flippers']
    #     dataSet = pd.read_csv("MushroomTrain.csv", header=None)
    #     dataset = np.array(dataSet.iloc[:, 0:-1]).tolist()
    #     label = ['shape', 'surface', 'color', 'bruises']
    return dataSet, label


def choostBestFeature(dataSet):
    num_feature = len(dataSet[0]) - 1
    base_entropy = ShannonEnt(dataSet)
    best_info_gain = 0.0
    for i in range(num_feature):
        feat_value = set([example[i] for example in dataSet])
        feat_entropy = 0.0
        for value in feat_value:
            sub_dataSet = splitDataset(dataSet, i, value)
            prob = float(len(sub_dataSet) / len(dataSet))
            feat_entropy += prob * ShannonEnt(sub_dataSet)
        info_gain = base_entropy - feat_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def labelOfLeaf(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_list:
            class_list[vote] = 0
        class_list[vote] += 1
    sorted_class_list = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_list[0][0]


def creatTree(dataSet, labels):
    class_list = [example[-1] for example in dataSet]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(dataSet[0]) == 1:
        return labelOfLeaf(class_list)
    best_feat = choostBestFeature(dataSet)
    best_feat_label = labels[best_feat]
    myTree = {best_feat_label: {}}
    unique_feat_values = set([example[best_feat] for example in dataSet])
    for value in unique_feat_values:
        myTree[best_feat_label][value] = creatTree(splitDataset(dataSet, best_feat, value), labels)
    return myTree


# def classify(inputTree, featLabels, testVec):
#     firstStr = inputTree.keys()[0]
#     secondDict = inputTree[firstStr]
#     featIndex = featLabels.index(firstStr)
#     for key in secondDict.keys():
#         if testVec[featIndex] == key:
#             if type(secondDict[key]).__name__ == 'dict':
#                 classLabel = classify(secondDict[key], featLabels, testVec)
#             else:
#                 classLabel = secondDict[key]
#     return classLabel

dataset, label = creatDataSet()
print(creatTree(dataset, label))


# classify(creatTree(dataset, label), label, [1, 1])