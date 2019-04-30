import numpy as np
from homework_1.bayes import Bayes
from homework_1.KNN import KNN

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

# create training_data using numpy
training_data = np.array(training_data)

# target_list is a list of the given data to predict.
target_list = [[155, 40, 35],
               [170, 70, 32],
               [175, 70, 35],
               [180, 90, 20]]

# target is the data you want to predict.
# The input form should be x, x, x. And x should be an integer.
target = input("Please input a target to predict(In the form of 155, 40, 35).\n")
target = [int(x) for x in target.split(",")]

# using KNN algorithm to predict the gender.
print("Implement KNN algorithm:")
# k is the number of nearest neighbor you want to select. Basically is 1, 3, 5 in this problem.
k = int(input("Please input the value of K(Could be 1, 3, 5 in this problem).\n"))
# age_included is used to control whether use the age feature or not. 1 means included and 0 is removed.
age_included = int(input("Please input the value of age_included(1 means age is included, 0 is not).\n"))
# initialize a object of KNN class, and pass the values you just input.
knn = KNN(k, age_included, target, training_data)
KNN.predict(knn)

# using Gaussian Naive Bayes algorithm to predict the gender.
print("Implement Naive Bayes algorithm:")
# age_included is used to control whether use the age feature or not. 1 means included and 0 is removed.
age_included = int(input("Please input the value of age_included(1 means age is included, 0 is not).\n"))
# initialize a object of Bayes class, and pass the values you just input.
bayes = Bayes(age_included, target, training_data)
Bayes.predict(bayes)

# also we can use a for loop to calculate all the target in the target_list
for tar in target_list:
    print("predicting: ", tar)
    knn_pre = KNN(k, age_included, tar, training_data)
    KNN.predict(knn_pre)
    bayes_pre = Bayes(age_included, tar, training_data)
    Bayes.predict(bayes_pre)
