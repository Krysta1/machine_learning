import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# First, load data
train_df = pd.read_csv('training.txt', sep=',', header=None,
                         names=["height", "weight", "age", "gender"])
test_df = pd.read_csv('test.txt', sep=',', header=None,
                        names=["height", "weight", "age", 'gender'])
print(train_df, test_df)
train_df.dropna(how='all', inplace=True)  # Delete missing values
test_df.dropna(how='all', inplace=True)
# change M to 0 and change W to 1
train_df.loc[train_df[(train_df['gender'] == 'M')].index, ['gender']] = 0
train_df.loc[train_df[(train_df['gender'] == 'W')].index, ['gender']] = 1
test_df.loc[test_df[(test_df['gender'] == 'M')].index, ['gender']] = 0
test_df.loc[test_df[(test_df['gender'] == 'W')].index, ['gender']] = 1

# Second: get X and Y
feature_names = ['height', 'weight', 'age']
train_X = train_df[feature_names].values
train_Y = train_df['gender'].values
y = train_Y  # the label
test_X = test_df[feature_names].values
test_Y = test_df['gender'].values

# Third: The label of classified
labels_type = np.unique(y)

# Fourth: compute Within-class scarrer matrix Sw
Sw = np.zeros([3, 3])
for i in range(0, 2):
    # Distinguishing data by label,y=0 and y=1
    xi = train_X[y == i]
    ui = np.mean(xi, axis=0)
    sw = ((xi - ui).T).dot(xi - ui)
    Sw = Sw + sw
print("Sw\n", Sw)

# Fifth: compute the Between-class scatter  SB
SB = np.zeros([3, 3])
U = np.zeros([3, 3])
U1 = np.zeros([3, 3])
u = np.mean(train_X, axis=0).reshape(3, 1)
for i in range(0, 2):
    n = train_X[y == i].shape[0]
    xii = train_X[y == i]
    u1 = np.mean(xii, axis=0).reshape(3, 1)
    sb = n * (u1 - u).dot((u1 - u).T)
    SB = SB + sb
print("SB\n", SB)

# sixth: compute w use Sw^-1*SB   eig特征向量，inv逆
vals, eigs = np.linalg.eig(np.linalg.inv(Sw).dot(SB))
# senventh: use the height and weight feature, make one low degree
w = np.vstack([eigs[:, 0], eigs[:, 1]])
transform_X = train_X.dot(w.T)  # changed 2 degrees features
print("transform\n", transform_X)
print("w\n", w)

# eighth: drow plot
labels_dict = train_df['gender'].unique()


def plot_LDA():
    ax = plt.subplot(111)  # only on chat
    for label, m, c in zip(labels_type, ['*', 'v'], ['red', 'black']):
        plt.scatter(transform_X[y == label][:, 0], transform_X[y == label][:, 1], c=c, marker=m, alpha=0.6, s=100,
                    label=labels_dict[label - 1])

    x1 = np.arange(0, 30, 3)
    x2 = (-w[0][1] - w[0][2] * x1) / w[1][0]  # 逻辑回归获取的回归系数，满足w0+w1*x1+w2*x2=0，即x2 =(-w0-w1*x1)/w2
    plt.plot(x1, x2)

    plt.xlabel('LDA1')
    plt.ylabel('LDA2')

    # 定义图例，loc表示的是图例的位置
    leg = plt.legend(loc='upper right', fancybox=True)
    # 设置图例的透明度为0.6
    leg.get_frame().set_alpha(0.6)
    # 坐标轴上的一簇簇的竖点
    plt.tick_params(axis='all', which='all', bottom='off', left='off', right='off', top='off', labelbottom='on',
                    labelleft='on')
    # 表示的是坐标方向上的框线
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.grid()
    plt.show()


plot_LDA()

# ninth:Test
m0 = np.mean(train_X[y == 0], axis=0)
m1 = np.mean(train_X[y == 1], axis=0)


def predict():
    return (test_X - 0.5 * (m0 + m1)).dot(w.T)


Y_predict = predict()
print('y prediction\n', Y_predict)
for i in range(4):
    if Y_predict[i][0] < 1:
        type = 'M'
    else:
        type = 'W'
    print(test_X[i], "  ", type)
