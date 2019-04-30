import numpy as np


class Bayes:
    """
            Author: Xinsheng Li
            Bayes is a class to implement Gaussian Naive Bayes algorithm.
            age_value: is used to control whether use the age feature or not.
            target_value_list: the data you want to predict.
            training data: as its name means.
        """
    def __init__(self, age_included, target_value_list, training_data):
        if age_included == 0:
            self.loop_range = 2
        else:
            self.loop_range = 3
        self.target_value_list = target_value_list
        self.training_data = training_data
        count = 0
        for x in self.training_data[:, 3]:  # calculate the priori probability of being a man or women.
            if x == 0:
                count += 1
        self.pro_women = count / len(self.training_data)
        self.pro_man = 1 - self.pro_women

    def calculate_probability(self, feature, x):
        """

        :param feature: to calculate every feature's mu and sigma.
        :param x: to calculate the probability be x under a Gaussian distribution.
        :return: the probability of being x.
        """
        mu = np.mean(feature)
        sigma = np.std(feature, ddof=1)
        print("The mu is %f and the sigma is %f." % (mu, sigma))  # output the mu and sigma.
        probability = (1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2)))
        # Gaussian Distribution
        return probability

    def calculate_man_pro(self, target_value_list):
        """

        :param target_value_list: the target we want to predict.
        :return: the probability of being a man.
        """
        data = self.training_data[self.training_data[:, 3] == 1]
        result = 1
        for i in range(self.loop_range):
            if i == 0:  # use str to record the feature name.
                str = "height"
            elif i == 1:
                str = "weight"
            else:
                str = "age"
            print("calculating mu and sigma of %s: " % str)
            result *= self.calculate_probability(data[:, i], target_value_list[i])
        print("The probability being a man is:", result * self.pro_man)
        return result * self.pro_man

    def calculate_women_pro(self, target_value_list):
        """

        :param target_value_list: the target we want to predict.
        :return: the probability of being a woman.
        """
        data = self.training_data[self.training_data[:, 3] == 0]
        result = 1
        for i in range(self.loop_range):
            if i == 0:  # use str to record the feature name.
                str = "height"
            elif i == 1:
                str = "weight"
            else:
                str = "age"
            print("calculating mu and sigma of %s: " % str)
            result *= self.calculate_probability(data[:, i],  target_value_list[i])
        print("probability being a women is :", result * self.pro_women)
        return result * self.pro_women

    def predict(self):
        """

        :return: the gender we predict.
        """
        result = int(self.calculate_man_pro(self.target_value_list) > self.calculate_women_pro(self.target_value_list))
        print("So we can predict that is a:", 'man' if result else 'woman')  # output the result.
        return result


