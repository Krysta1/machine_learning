from math import *

MODE = 'MAX'  # a variable to control the way select two clusters(MIN for single linkage and MAX for complete linkage)
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


# calculate two lists' Euclidean distance.
def cal_distance(vec1, vec2):
    tmp = 0
    for i in range(len(vec1)):
        tmp += (vec1[i] - vec2[i]) ** 2
    return sqrt(tmp)


# initialize the distance between each two samples
# return a dictionary the key is (X, X) and the value is the distance.
def init_distance_dic():
    distance_dic = {}
    for i in range(len(training_data)):
        for j in range(i + 1, len(training_data)):
            s = (i, j)
            distance_dic[s] = cal_distance(training_data[i][:3], training_data[j][:3])
    return distance_dic


# get the minimum distance in the current tree and select the minimum one
# return the minimum distance and two nodes which including the two examples
def min_distance(current_tree, dic, step):
    min_dis = inf
    for i in range(len(current_tree)):  # loop all the clusters in the current tree
        for j in range(i + 1, len(current_tree)):
            for x in current_tree[i]:  # loop all the samples in each clusters
                for y in current_tree[j]:
                    x1, x2 = min(x, y), max(x, y)  # keep the smaller index in the front because the dictionary format
                    if dic[(x1, x2)] < min_dis:  # save the smaller distance
                        min_dis = dic[(x1, x2)]
                        tmp1, tmp2 = i, j
                        first_index, second_index = x1, x2
    print("The {}th step: The minimum distance is {} between {} and {}".format(step, min_dis, first_index, second_index))
    print("Merge {} {}".format(current_tree[tmp1], current_tree[tmp2]))
    return min_dis, current_tree[tmp1], current_tree[tmp2]


# get the maximum distance in the current tree and select the minimum one
# return the minimum distance and two nodes which including the two examples
def max_distance(current_tree, dic, step):
    min_dis = inf
    for i in range(len(current_tree)):
        for j in range(i + 1, len(current_tree)):
            max_dis = -inf
            for x in current_tree[i]:
                for y in current_tree[j]:
                    x1, x2 = min(x, y), max(x, y)
                    if dic[(x1, x2)] > max_dis:
                        max_dis = dic[(x1, x2)]
            if max_dis < min_dis:
                min_dis = max_dis
                first_index, second_index = x, y
                first_node, second_node = current_tree[i], current_tree[j]
    print("The {}th step: The minimum distance is {} between {} and {}".format(step, min_dis, first_index, second_index))
    print("Merge {} {}".format(first_node, second_node))

    return min_dis, first_node, second_node


if __name__ == "__main__":
    hierarchy_tree = [[x] for x in range(14)]
    # [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13]]
    distance_dic = init_distance_dic()
    step = 0
    if MODE == "MIN":
        print("Single linkage...............")
    else:
        print("Complete linkage...............")
    while True:
        step += 1
        if len(hierarchy_tree) == 2:
            break
        if MODE == "MIN":
            minimum_dis, first_node, second_node = min_distance(hierarchy_tree, distance_dic, step)
        else:
            minimum_dis, first_node, second_node = max_distance(hierarchy_tree, distance_dic, step)

        hierarchy_tree.remove(first_node)
        hierarchy_tree.remove(second_node)

        tmp = list(set(first_node) | set(second_node))
        hierarchy_tree.append(tmp)
        print(hierarchy_tree)

