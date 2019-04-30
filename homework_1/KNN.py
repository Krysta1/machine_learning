from collections import Counter
from math import *


class KNN:
    """
        Author: Xinsheng Li
        KNN is a class to implement KNN algorithm.
        k_value: the number of nearest neighbor you want to select.
        age_value: is used to control whether use the age feature or not.
        target_value_list: the data you want to predict.
        training data: as its name means.
    """
    def __init__(self, k_value, age_included, target_value_list, training_data):
        if age_included == 0:
            self.loop_range = 2  # use to control whether use the age feature or not
        else:
            self.loop_range = 3
        self.k = k_value
        self.target = target_value_list
        self.training_data = training_data
        self.result = [x for x in training_data[:, 3]]  # result is all the label list.

    def distance_calculation(self, target, training_data):
        distance = []  # a list to save the distance.
        for data in training_data:  # calculate the Cartesian Distance and save into distance list.
            tmp = 0
            for i in range(self.loop_range):
                tmp = tmp + (data[i] - target[i]) ** 2
            distance.append(sqrt(tmp))
        for j in range(len(distance)):  # print the distance between target and the point.
            print("The distance between Point %s and target is: %f" % (self.training_data[j], distance[j]))
        return distance

    def neighbor_selection(self, distance, k):  # select the k nearest neighbors in the list.
        tmp = sorted(list(zip(distance, self.result)))
        gender_list = []
        point_select = []
        for i in range(k):
            gender_list.append(tmp[i][1])
            point_select.append(tmp[i][0])
        print("We get the %d nearest neighbors' distance is: %s" % (k, point_select))
        print("We get the %d nearest neighbors' gender is: %s" % (k, gender_list))
        return gender_list

    def predict(self):  # predict the gender.
        d = Counter(self.neighbor_selection(self.distance_calculation(self.target, self.training_data), self.k))
        print("So we can predict that is a:", 'man' if d.most_common()[0][0] else 'woman')
        return d.most_common()[0][0]

