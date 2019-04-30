import csv
import math

# import numpy as np

NUMBER = 100
attributes = [["b", "c", "x", "f", "k", "s"], ["f", "g", "y", "s"],
              ["n", "b", "c", "g", "r", "p", "u", "e", "w", "y", "t", "f"], ["t", "f"]]
attribute = ["shape", "surface", "color", "bruise"]

label_value = ["e", "p"]
shape_value = ["b", "c", "x", "f", "k", "s"]
surface_value = ["f", "g", "y", "s"]
color_value = ["n", "b", "c", "g", "r", "p", "u", "e", "w", "y", "t", "f"]
bruise_value = ["t", "f"]
# odor_value=["a","l","c","y","f","m","n","p","s"]
# 读取csv至字典
csvFile = open("MushroomTrain.csv", "r")
reader = csv.reader(csvFile)
# 建立空字典
result = []
for item in reader:
    result.append(item[0:-1])


def entropy(n):
    if n != 0:
        return -n * math.log(n, 2)
    else:
        return 0


def get_gain(gain, num, e, p):
    for i in range(len(num)):
        if num[i] != 0:
            gain = gain - (entropy(e[i] / num[i]) + entropy(p[i] / num[i])) * num[i] / sum(num)
    return gain


def get_gainratio(gain, num, e, p):
    base = 0
    x = 0
    for i in range(len(num)):
        if num[i] != 0:
            gain = gain - (entropy(e[i] / num[i]) + entropy(p[i] / num[i])) * num[i] / sum(num)
            base = base + entropy(num[i] / sum(num))
            x = x + 1
    if x <= 2:
        return gain
    else:
        gainratio = gain / base
        return gainratio


def get_attri_e_p(sets, attr_index):
    num = [0] * len(attributes[attr_index])
    e = [0] * len(attributes[attr_index])
    p = [0] * len(attributes[attr_index])
    for i in sets:
        for j in range(len(attributes[attr_index])):  # attr_index=1:6
            if result[i][attr_index + 1] == attributes[attr_index][j]:
                num[j] = num[j] + 1
                if result[i][0] == "e":
                    e[j] = e[j] + 1
                else:
                    p[j] = p[j] + 1
                continue
    return num, e, p


def divid_set(sets, attri_index):
    list_sets = []
    for j in range(len(attributes[attri_index])):
        each_attribute_sets = []
        for i in sets:
            if result[i][attri_index + 1] == attributes[attri_index][j]:
                each_attribute_sets.append(i)
        list_sets.append(each_attribute_sets)
    return list_sets


def get_label_entro(sets):
    label = [0, 0]
    for item in sets:
        for i in range(len(label_value)):
            if result[item][0] == label_value[i]:
                label[i] = label[i] + 1
    gain = entropy(label[0] / len(sets)) + entropy(label[1] / len(sets))
    return gain


def generate_tree(gain, sets, attribute_set):
    tree = {}
    gainratios = []
    gains = []
    # if len(attribute_set)==0:
    for i in range(len(attribute_set)):
        num, e, p = get_attri_e_p(sets, i)
        gainratios.append(get_gainratio(gain, num, e, p))
        gains.append(get_gain(gain, num, e, p))
    generate_attri_index = list(attribute_set)[gainratios.index(max(gainratios))]
    list_sets = divid_set(sets, generate_attri_index)
    tree[attribute[generate_attri_index]] = {}
    attribute_set.remove(generate_attri_index)
    this_attribute_set = attribute_set.copy()
    for i in range(len(list_sets)):
        if list_sets[i] != [] and len(attribute_set) != 0:
            tree[attribute[generate_attri_index]][attributes[generate_attri_index][i]] = generate_tree(
                get_label_entro(list_sets[i]), list_sets[i], attribute_set)
        elif list_sets[i] != [] and len(attribute_set) == 0:
            tree[attribute[generate_attri_index]][attributes[generate_attri_index][i]] = result[list_sets[i][0]][0]
        attribute_set = this_attribute_set.copy()
    return tree


l = [i for i in range(len(result))]
sets = set(l)
label = [0, 0]
for item in result:
    for i in range(len(label_value)):
        if item[0] == label_value[i]:
            label[i] = label[i] + 1
gain = entropy(label[0] / len(result)) + entropy(label[1] / len(result))
attribute_sets_number = {0, 1, 2, 3}
dict_tree = generate_tree(gain, sets, attribute_sets_number)
print(dict_tree)

csvFile.close()
print(result)

