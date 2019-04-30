import numpy as np
import pandas as pd
from adaboost.Adaboost import *
from adaboost.WeakClassifier import *
from adaboost.Plot2D import *


# # data.csv is created by make_data.py
# data = pd.read_csv('data.csv')
#
# # get X and y
# X = data.iloc[:, :-1].values
# y = data.iloc[:, -1].values
data = np.loadtxt("train.txt", delimiter=",")
X = data[:, :-1]
y = data[:, -1]
y[y==0] = -1
# print(X, y)
# train the AdaboostClassifier
clf = AdaboostClassifier()
times, alpha, weak_classifier, w = clf.fit(X, y)

# plot original data
# Plot2D(data).pause(3)

# plot Adaboost decision_threshold
# for i in range(times):
#     if clf.weak[i].decision_feature == 0:
#         plt.plot([clf.weak[i].decision_threshold, clf.weak[i].decision_threshold], [0, 8])
#     else:
#         plt.plot([0, 8], [clf.weak[i].decision_threshold, clf.weak[i].decision_threshold])
# plt.pause(3)
test_data = np.loadtxt("test.txt", delimiter=",")
test_X = test_data[:, :-1]
test_y = test_data[:, -1]
test_y[test_y == 0] = -1
# print(X, y)
# result
# cnt = 0
# for i in range(len(result)):
#     if int(result[i]) == int(test_y[i]):
#         cnt += 1
# print("Accuracy is %f" % (cnt / len(result)))
print(alpha, weak_classifier)
# print(weak_classifier[0].pred)
wk = WeakClassifier()
ret = np.array([0.0] * len(test_y))
for i in range(len(alpha)):
    print(weak_classifier[i].W)
    weak_classifier[i].pred = wk.fit(test_X, test_y, weak_classifier[i].W)
    # print(weak_classifier[i].pred)
    ret += alpha[i] * weak_classifier[i].pred
result = np.sign(ret)
cnt = 0
for i in range(len(test_y)):
    if int(result[i]) != int(test_y[i]):
        cnt += 1
print(cnt)
print("Error rate is %f" % (cnt / len(test_y)))