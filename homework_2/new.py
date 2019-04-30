import operator
import pandas as pd
import numpy as np

labels = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor']


def loadData(filepath):
    Data = pd.read_csv(filepath)
    data = np.array(Data).tolist()
    return data

# calculate the information entropy of dataset
def cla_entropy(dataSet):
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
        shannonEnt -= prob * np.log2(prob)
    return shannonEnt


#
def split_data(dataSet, index, value):
    newDataSet = []
    for item in dataSet:
        if item[index] == value:
            reducedFeatVec = item[:index]
            reducedFeatVec.extend(item[index + 1:])
            newDataSet.append(reducedFeatVec)
    return newDataSet


def best_feat(dataSet):
    numFeature = len(dataSet[0])
    baseEntropy = cla_entropy(dataSet)
    infoGain = 0.0
    bestFeature = -2
    for k in range(1, numFeature):
        featureList = [x[k] for x in dataSet]
        uniqueFeatureList = set(featureList)
        newEntropy = 0.0
        for value in uniqueFeatureList:
            subDataSet = split_data(dataSet, k, value)
            sub_DS_prob = float(len(subDataSet) / len(dataSet))
            newEntropy += sub_DS_prob * cla_entropy(subDataSet)
        Gain = baseEntropy - newEntropy
        if Gain > infoGain:
            infoGain = Gain
            bestFeature = k
    return bestFeature


def majority(class_list):
    classCount = {}
    for vote in class_list:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    print('sortedClassCount[0][0]', sortedClassCount[0][0])
    return sortedClassCount[0][0]


def build_tree(dataSet, labels):
    classList = [j[0] for j in dataSet]
    # First stop Condition:
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # Second stop Condition:
    if len(dataSet[0]) == 1:
        return majority(classList)
    bestFeat = best_feat(dataSet)
    bestFeatLabel = labels[bestFeat]
    desiciontree = {bestFeatLabel: {}}
    labels.remove(bestFeatLabel)
    featureValue = [z[bestFeat] for z in dataSet]
    uniqfeatureValue = set(featureValue)
    for value in uniqfeatureValue:
        sublabels = labels[:]
        desiciontree[bestFeatLabel][value] = build_tree(split_data(dataSet, bestFeat, value), sublabels)
    return desiciontree


def classify(tree, labels, testVec):
    firstStr = list(tree.keys())[0]  # Get the first node of the decision tree, Odor
    secondFloor = tree[firstStr]  # Get the second node of the decision tree, 'a': 'e', 'p': 'p', 'l': 'e', 'n': 'e'
    Index = labels.index(firstStr)  # Get index of Odor
    for key in secondFloor.keys():  # 'a', 'p', 'l', 'n'
        if testVec[Index] == key:
            if type(secondFloor[key]).__name__ == 'dict':
                class_label = classify(secondFloor[key], labels, testVec)
            else:
                class_label = secondFloor[key]
    return class_label


def main():
    training_data = loadData('MushroomTrain.csv')
    test_data = loadData('MushroomTest.csv')
    print('The Root Feature we choose:\n', labels[best_feat(test_data)])
    decision_tree = build_tree(training_data, labels[:])
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
