import operator
import csv
import math
import pandas as pd
import numpy as np

labels = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor']
# load training data and test data from .csv file


def loadData(filepath):
    # datafile = csv.reader(open(filepath))
    # dataSet = []
    # for item in datafile:
    #     dataSet.append(item)
    training_data = pd.read_csv(filepath)
    dataSet = np.array(training_data)
    return dataSet


def loadTestSet(fileName):
    testfile = csv.reader(open(fileName))
    testSet = []
    for item in testfile:
        testSet.append(item)
    return testSet


# calculate the information entropy of dataset
def calcShannonEnt(dataSet):
    lenDataSet = len(dataSet)
    shannonEnt = 0.0
    classCount = {}
    for item in dataSet:
        currentClass = item[0]
        if currentClass not in classCount.keys():
            classCount[currentClass] = 0
        classCount[currentClass] += 1
    for key in classCount:
        prob = float(classCount[key]) / lenDataSet
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt


#
def splitDataSet(dataSet, index, value):
    newDataSet = []
    for item in dataSet:
        if item[index] == value:
            reducedFeatVec = item[:index]
            reducedFeatVec.extend(item[index + 1:])
            newDataSet.append(reducedFeatVec)
    return newDataSet


def chooseBestFeature(dataSet):
    numFeature = len(dataSet[0])
    baseEntropy = calcShannonEnt(dataSet)
    infoGain = 0.0
    bestFeature = -2
    for k in range(1, numFeature):
        featureList = [x[k] for x in dataSet]
        uniqueFeatureList = set(featureList)
        newEntropy = 0.0
        for value in uniqueFeatureList:
            subDataSet = splitDataSet(dataSet, k, value)
            sub_DS_prob = float(len(subDataSet) / len(dataSet))
            newEntropy += sub_DS_prob * calcShannonEnt(subDataSet)
        Gain = baseEntropy - newEntropy
        if Gain > infoGain:
            infoGain = Gain
            bestFeature = k
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    print('sortedClassCount[0][0]', sortedClassCount[0][0])
    return sortedClassCount[0][0]


def buildTree(dataSet, labels):
    classList = [j[0] for j in dataSet]
    # First stop Condition:
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # Second stop Condition:
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeature(dataSet)
    bestFeatLabel = labels[bestFeat]
    desicionTree = {bestFeatLabel: {}}
    labels.remove(bestFeatLabel)
    featureValue = [z[bestFeat] for z in dataSet]
    uniqfeatureValue = set(featureValue)
    for value in uniqfeatureValue:
        subLabels = labels[:]
        desicionTree[bestFeatLabel][value] = buildTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return desicionTree


def classify(Tree, Labels, testVec):
    firstStr = list(Tree.keys())[0]  # Get the first node of the decision tree, Odor
    secondFloor = Tree[firstStr]  # Get the second node of the decision tree, 'a': 'e', 'p': 'p', 'l': 'e', 'n': 'e'
    Index = Labels.index(firstStr)  # Get index of Odor
    for key in secondFloor.keys():  # 'a', 'p', 'l', 'n'
        if testVec[Index] == key:
            if type(secondFloor[key]).__name__ == 'dict':
                classLabel = classify(secondFloor[key], Labels, testVec)
            else:
                classLabel = secondFloor[key]
    return classLabel


def main():
    training_data = loadData('MushroomTrain.csv')
    test_data = loadTestSet('MushroomTest.csv')
    print('The Root Feature we choose:\n', labels[chooseBestFeature(test_data)])
    decision_tree = buildTree(training_data, labels[:])
    print(decision_tree)
    count = 0
    for test_item in test_data:
        print('Classifying :', test_item)
        result = classify(decision_tree, labels, test_item)
        print('The true label is: ', test_item[0])
        print('The classify result is:%s' % result)
        if test_item[0] == result:
            count += 1
    print("The accuracy of test data is %f." % (count / len(test_data)))

    count = 0
    for training_item in training_data:
        result = classify(decision_tree, labels, training_item )
        if training_item[0] == result:
            count += 1
    print("The accuracy of training data is %f." % (count / len(training_data)))


if __name__ == '__main__':
    main()
