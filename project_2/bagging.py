from project_2.logistic_reg import *
from collections import Counter

lg = Logistic(learning_rate=0.1, num_iter=100)
num_of_bags = 100


# logistic using sampled data and predict the result using bagging method.
def bagging_classify():
    print("Start %d times bagging: " % num_of_bags)
    for i in range(num_of_bags):
        print("------------------------------------")
        print("%d -th iteration: " % (i + 1))
        X, y = lg.sampled_data("train.txt")  # get sampled data
        X, y = transfer(X, y)  # transfer the format wanted
        bagging_result = []
        theta = lg.gradient(X, y)
        gender_list = lg.predict(test_X, test_y, theta, True)
        bagging_result.append(gender_list)  # save all the prediction results into a list

    bagging_result = np.array(bagging_result).T  # make a transfer to use Counter
    final_result = []
    for x in bagging_result[0]:
        # Using Counter to return the most common result of bagging
        final_result.append(Counter(x).most_common(1)[0][0])

    count = 0
    # count the number of wrong prediction
    for i in range(len(test_y)):
        if int(final_result[i]) != test_y[i]:
            count += 1
    print("****************************************")
    print("Final prediction error number: %d" % count)
    print("Final error rate: %f" % (1.0 * count / len(test_y)))


# Single logistic classified and the predict result.
def single_classify():
    single_X, single_y = lg.load_data("train.txt")
    single_X, single_y = transfer(single_X, single_y)
    theta = lg.gradient(single_X, single_y)
    gender_list = lg.predict(test_X, test_y, theta)

    count = 0
    for i in range(len(test_y)):
        if int(gender_list[i]) != test_y[i]:
            count += 1

    print("Single logistic error number: %d" % count)
    print("Single logistic rate: %f" % (1.0 * count / len(test_y)))


if __name__ == "__main__":
    test_X, test_y = lg.load_data("test.txt")
    test_X, test_y = transfer(test_X, test_y)

    print("------------------------------------")
    print("Start single logistic classify:")
    single_classify()
    bagging_classify()

