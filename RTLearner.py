'''
An implementation of the Random Forest Trees.
Made for the Assess learners Project for CS7464 - ML4T
'''

import numpy as np

class RTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def author(self):
        return 'rsadiq3'

    def addEvidence(self, dataX, dataY):
        combined_data = np.column_stack((dataX, dataY)) # combine X and Y into one array for ease of manipulation iin build_tree
        self.dtree = self.build_tree(combined_data)

    def build_tree(self, data):
        if np.unique(data[:, -1]).size == 1: # check if all y data are the same
            return np.array([['leaf', np.mean(data[:, -1]), np.nan, np.nan]])
        if data.shape[0] <= self.leaf_size:
            if data.shape[0] != 0:
                return np.array([['leaf', np.mean(data[:, -1]), np.nan, np.nan]])
            else:
                return np.array([['leaf', '0.0', np.nan, np.nan]])
        else:
            np.set_printoptions(threshold=np.nan)
            np.seterr(divide='ignore', invalid='ignore')
            x_columns = data.shape[1] - 1 # count the number of X columns available in the data set
            random_rows, best_feature_index = self.generate_random_rows(x_columns, data)
            splitVal = np.mean(random_rows[:, best_feature_index])
            left_tree = self.build_tree(data[data[:, best_feature_index] <= splitVal])
            right_tree = self.build_tree(data[data[:, best_feature_index] > splitVal])
            root = np.array([best_feature_index, splitVal, 1, left_tree.shape[0] + 1])
            return np.vstack((root, left_tree, right_tree))

    def generate_random_rows(self, max_value, data):
        best_feature_value = np.random.randint(max_value, size=1)
        best_feature_index = best_feature_value[0]
        random_rows = data[np.random.randint(data.shape[0], size=2), :]
        return random_rows, best_feature_index

    def query(self, dataX):
        if self.dtree[0][0] == 'leaf':
            Y = [self.dtree[0][1] for i in range(dataX.shape[0])]
            return np.array(Y)
        else:
            Y = []
            for data in dataX:
                node = 0
                while node < self.dtree.shape[0] and self.dtree[node][0] != 'leaf':
                    factor = int(float(self.dtree[node][0]))
                    if data[factor] > float(self.dtree[node][1]):
                        node += int(float(self.dtree[node][3]))
                    else:
                        node += int(float(self.dtree[node][2]))
                Y.append(float(self.dtree[node][1]))
            return np.array(Y)


if __name__=="__main__":
    print "this is the RT Learner"
