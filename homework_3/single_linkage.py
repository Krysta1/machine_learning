import collections
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

mode = 'MIN'


def cal_distance(vec1, vec2):
    tmp = 0
    for i in range(len(vec1)):
        tmp += (vec1[i] - vec2[i]) ** 2
    return sqrt(tmp)


# print(cal_distance(training_data[0][:3], training_data[3][:3]))

distance_dic = collections.OrderedDict()

for i in range(len(training_data)):
    for j in range(i + 1, len(training_data)):
        s = (i, j)
        distance_dic[s] = cal_distance(training_data[i][:3], training_data[j][:3])
sorted_distance = sorted(distance_dic.items(), key=lambda x: x[1])
# [((3, 6), 1.0), ((9, 11), 3.605551275463989), ((10, 13), 3.605551275463989), ((8, 12), 5.0990195135927845)......]
if mode == 'MAX':
    set_list = []
    for node, distance in sorted_distance:
        flag = 0
        tmp1, tmp2 = None, None
        i_in_x, j_in_x = False, False
        i, j = node
        for x in set_list:
            # print(x)
            if i in x and j in x:
                flag = 3
            else:
                if i in x:
                    i_in_x = True
                    tmp1 = x
                    flag += 1
                if j in x:
                    j_in_x = True
                    tmp2 = x
                    flag += 1

        if flag == 0:
            print("Merge {} {}".format(i, j))
            set_list.append({i, j})

        if flag == 1:
            if i_in_x:
                print("Merge {} {}".format(tmp1, node))
                set_list.remove(tmp1)
                tmp1.add(j)
                set_list.append(tmp1)

            if j_in_x:
                print("Merge {} {}".format(node, tmp2))
                set_list.remove(tmp2)
                tmp2.add(i)
                set_list.append(tmp2)

        if flag == 2:
            print("Merge {} {}".format(tmp1, tmp2))
            set_list.remove(tmp1)
            set_list.remove(tmp2)
            tmp = tmp1 | tmp2
            set_list.append(tmp)

        if flag == 3:
            continue

        for x in sorted_distance:
            if (i, j) == x[0]:
                min_distance = x[1]

        print("The minimum distance is {}".format(min_distance))

        print("After merging: {}".format(set_list))
        if len(set_list[0]) == 14:
            print("..................")
            print("Finish clustering")
            break


if mode == "MIN":
    set_list = []
    for node, distance in sorted_distance:
        flag = 0
        tmp1, tmp2 = None, None
        i_in_x, j_in_x = False, False
        i, j = node
        for x in set_list:
            # print(x)
            if i in x and j in x:
                flag = 3
            else:
                if i in x:
                    i_in_x = True
                    tmp1 = x
                    flag += 1
                if j in x:
                    j_in_x = True
                    tmp2 = x
                    flag += 1

        if flag == 0:
            print("Merge {} {}".format(i, j))
            set_list.append({i, j})

        if flag == 1:
            if i_in_x:
                print("Merge {} {}".format(tmp1, node))
                set_list.remove(tmp1)
                tmp1.add(j)
                set_list.append(tmp1)

            if j_in_x:
                print("Merge {} {}".format(node, tmp2))
                set_list.remove(tmp2)
                tmp2.add(i)
                set_list.append(tmp2)

        if flag == 2:
            print("Merge {} {}".format(tmp1, tmp2))
            set_list.remove(tmp1)
            set_list.remove(tmp2)
            tmp = tmp1 | tmp2
            set_list.append(tmp)

        if flag == 3:
            continue

        for x in sorted_distance:
            if (i, j) == x[0]:
                min_distance = x[1]

        print("The minimum distance is {}".format(min_distance))

        print("After merging: {}".format(set_list))
        if len(set_list[0]) == 14:
            print("..................")
            print("Finish clustering")
            break
