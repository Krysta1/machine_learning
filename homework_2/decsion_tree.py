import operator
from math import log

# 计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  # 计算实例的总数
    labelCounts = {}  # 创建空字典
    for featVec in dataSet:  # 为所有可能分类创建字典，字典的键表示类别，值表示该类别出现的次数
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():  # 如果当前标签不在字典中，就扩展字典，并将该标签加入字典
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # 否则，当前类别标签出现的次数+1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 计算类别出现的频率
        shannonEnt -= prob * log(prob, 2)  # 用此概率，以2为底求对数
    return shannonEnt

def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

myDate, labels = createDataSet()
print("原始数据集：", myDate)
shannonEnt = calcShannonEnt(myDate)
print("香农熵：", shannonEnt)

def splitDataSet(dataSet, axis, value):  # 待划分的数据集、划分数据集特征、需要返回的特征的值
    retDataSet = []  # 创建新的list对象
    for featVec in dataSet:  # 抽取
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 计算特征属性的个数
    baseEntropy = calcShannonEnt(dataSet)  # 调用前面写好的calcShannonEnt函数，计算整个数据集的原始香农熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  # 将所有可能的特征值写入新的list中
        uniqueVals = set(featList)  # 使用Python语言原生的集合set数据类型，从类表中得到类表中唯一元素值
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)  # 调用前面写好的splitDataSet函数，对每个属性值划分一次数据集
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)  # 计算数据集的新熵，并求和
        infoGain = baseEntropy - newEntropy  # 求信息增益
        if infoGain > bestInfoGain:  # 比较所有特征的信息增益，
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature  # 返回最好特征划分的索引值

def majorityCnt(classList):
    classCount = {}  # 创建唯一值的数据字典，用于存储每个类标签出现的频率
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1),
                              reverse=True)  # 利用operator操作键值排序字典，并返回出现次数最多的分类名称
    return sortedClassCount[0][0]


# 创建树的函数代码
def createTree(dataSet, labels): # 数据集和标签列表
    classList = [example[-1] for example in dataSet]  # 获取数据集的标签（数据集每条记录的最后列数据）
    # 递归停止的第一个条件
    if classList.count(classList[0]) == len(classList): # 类别完全相同就停止继续划分数据
        return classList[0]
    # 递归停止的第二个条件
    if len(dataSet[0]) == 1: # 遍历完所有特征时返回出现次数最多的类别（无法简单地返回唯一的类标签，使用前面的多数表决方法）
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet) # 找出最佳数据集划分的特征属性的索引
    bestFeatLabel = labels[bestFeat]  # 获取最佳索引对应的值
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat]) # 删除最佳索引的列
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)  # 使用set集合去除重复的值，得到列表包含的所有属性值
    for value in uniqueVals:  # 遍历所有最佳划分的分类标签
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels) # 递归调用
    return myTree

myDat, labels = createDataSet()
myTree = createTree(myDat, labels)
print("myTree:",myTree)