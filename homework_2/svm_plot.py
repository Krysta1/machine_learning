from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np

data = np.array([[1, 2, -1], [2, 3, 1], [2, 1, -1], [3, 4, 1], [1, 3, -1], [4, 4, 1]])
X = data[:, 0:-1]
y = data[:, -1]


def plot_decision_boundary(pred_func):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

    xx = np.linspace(0, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    b = clf.support_vectors_[0]
    yy_down = a * xx + (b[1] - a * b[0])
    b = clf.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a * b[0])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.plot(xx, yy)
    plt.plot(xx, yy_down)
    plt.plot(xx, yy_up)
    plt.show()


clf = svm.SVC(C=1000, kernel='linear')
clf.fit(X, y)

w = clf.coef_[0]
a = -w[0] / w[1]
b = (clf.intercept_[0]) / w[1]

plot_decision_boundary(lambda x: clf.predict(x))

print("W:", w)
print("a:", a)
print("b:", b)
print("support_vectors_:", clf.support_vectors_)
print("clf.coef_:", clf.coef_)
