import numpy as np
import random
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


# 不用kd树的KNN
class Model():
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.k = len(X_train[0])                 # 特征数量
        self.class_num = len(set(y_train))       # 标签的种类


    def distance(self, node, x):
        return sum((node - x) ** 2)          # 欧式距离


    def knn_single_predict(self, x, k):
        '''
        预测单个目标点的标签
        :param x: 单个目标点
        :param k: k近邻法的k
        :return: 预测的目标点的标签
        '''
        nearest = []                            # 用来保存k个最近邻的列表，（距离，点，对应的标签）,不能用字典，可能会有重复点
        nearestDistance = float('inf')          # 当前k个最近邻中的最远距离

        for i in range(len(self.X_train)):
            if len(nearest) < k or self.distance(self.X_train[i], x) < nearestDistance:
                nearest.append((self.distance(self.X_train[i], x), self.X_train[i], self.y_train[i]))
                if len(nearest) > k:
                    max_index = nearest.index(max(nearest, key = lambda x: x[0]))
                    del nearest[max_index]
                nearestDistance = max(nearest, key = lambda x: x[0])[0]

        knodes = [(i[1], i[2]) for i in nearest]                # k个最近邻点，（X，y)
        class_count = [0 for _ in range(self.class_num)]        # 计数器，索引对应标签y
        for node in knodes:
            class_count[node[1]] += 1
        return class_count.index(max(class_count))        # 预测的y


    def knn_predict(self, X_test, k):
        '''
        对测试集的预测
        :param X_test: 测试集, np.array, (n, self.k)
        :param k: k近邻法的k
        :return: 对测试集的预测y_pred, np.array, (n, )
        '''
        return np.array([self.knn_single_predict(x, k) for x in X_test])


    def accuracy(self, X_test, y_test, k):
        '''预测精确度'''
        return np.mean(self.knn_predict(X_test, k) == y_test)


if __name__ == '__main__':
    iris = load_iris()
    X = np.array(iris.data)
    y = np.array(iris.target)

    random.seed(0)
    index = random.sample(range(0, 150), 150)
    X_train, X_test, y_train, y_test = X[index[: 120]], X[index[120: ]], y[index[: 120]], y[index[120:]]

    fit = Model(X_train, y_train)
    y_pred = fit.knn_predict(X_test, k = 5)
    accuracy = fit.accuracy(X_test, y_test, k = 5)
    print('the accuracy of predicting iris by knn without kdtree:', accuracy)
    # 注意这里和用kdtree的knn结果可能不一样，因为有些点的距离相等，kdtree和普通方法的遍历方式不一样



