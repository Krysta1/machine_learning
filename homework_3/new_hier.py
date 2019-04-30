from math import *

training_data = [[170, 57, 32, 0],
                 [192, 95, 28, 1],
                 [150, 45, 30, 0],
                 [170, 65, 29, 1],
                 [175, 78, 35, 1],
                 [185, 90, 32, 1],
                 [170, 65, 28, 0],
                 [155, 48, 31, 0],
                 [160, 55, 30, 0],
                 [182, 80, 30, 1],
                 [175, 69, 28, 0],
                 [180, 80, 27, 1],
                 [160, 50, 31, 0],
                 [175, 72, 30, 1]]

MODE = 'MIN'


def cal_distance(vec1, vec2):
    tmp = 0
    for i in range(len(vec1)):
        tmp += (vec1[i] - vec2[i]) ** 2
    return sqrt(tmp)


distance_dic = {}

for i in range(len(training_data)):
    for j in range(i + 1, len(training_data)):
        s = (i, j)
        distance_dic[s] = cal_distance(training_data[i][:3], training_data[j][:3])
print(distance_dic)

hierarchy_tree = [[x] for x in range(14)]
# [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13]]


def min_distance(current_tree):
    min_dis = inf
    for i in range(len(current_tree)):
        for j in range(i + 1, len(current_tree)):
            for x in current_tree[i]:
                for y in current_tree[j]:
                    x1, x2 = min(x, y), max(x, y)
                    if distance_dic[(x1, x2)] < min_dis:
                        min_dis = distance_dic[(x1, x2)]
                        tmp1, tmp2 = i, j
    return min_dis, current_tree[tmp1], current_tree[tmp2]
# tree = [[0], [1], [2], [3, 6], [4], [5], [7], [8], [9, 11], [10, 13], [12]]
# tree = [[1, 5], [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13]]
# print(min_distance(tree)


def max_distance(current_tree):
    min_dis = inf
    for i in range(len(current_tree)):
        for j in range(i + 1, len(current_tree)):
            max_dis = -inf
            for x in current_tree[i]:
                for y in current_tree[j]:
                    x1, x2 = min(x, y), max(x, y)
                    if distance_dic[(x1, x2)] > max_dis:
                        max_dis = distance_dic[(x1, x2)]
                        # tmp1, tmp2 = x1, x2

            if max_dis < min_dis:
                min_dis = max_dis
                first_node, second_node = current_tree[i], current_tree[j]

    return min_dis, first_node, second_node

# tree = [[2, 7, 0, 8, 12], [1, 5, 3, 6, 10, 13, 4, 9, 11]]
# tree = [[0], [1], [2], [3, 6], [4], [5], [7], [8], [9], [10], [11], [12], [13]]
# tree = [[1, 5], [2, 7], [0, 8, 12], [3, 6, 10, 13, 4, 9, 11]]
# tree = [[1, 5], [2, 7, 0, 8, 12], [3, 6, 10, 13, 4, 9, 11]]
# print(max_distance(tree))


while True:
    if len(hierarchy_tree) == 2:
        break
    if MODE == "MIN":
        minimum_dis, first_index, second_index = min_distance(hierarchy_tree)
    else:
        minimum_dis, first_index, second_index = max_distance(hierarchy_tree)

    hierarchy_tree.remove(first_index)
    hierarchy_tree.remove(second_index)

    tmp = list(set(first_index) | set(second_index))
    hierarchy_tree.append(tmp)
    print(hierarchy_tree)
