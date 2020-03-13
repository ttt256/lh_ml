import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


# 多分类的朴素贝叶斯模型（特征均为连续型变量，且关于y的条件分布为高斯分布）
# 先定义一个高斯分布密度函数的类
class GaussianFunc():
    def gaussian_pdf(self, x, mu, sigma):
        '''
        :return: 高斯分布的密度函数
        '''
        return np.exp(- (x - mu) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * math.pi) * sigma)

    def pdfFunc(self, X):
        '''
        :param X: np.array，(n, m)，训练集特征
        :return: 返回一个存储着高斯分布密度函数的列表
        '''
        mu = X.mean(0)
        sigma = X.std(0)
        def func(i):
            def helper(x):
                return self.gaussian_pdf(x, mu[i], sigma[i])
            return helper
        return [func(i) for i in range(X.shape[1])]


# 高斯型朴素贝叶斯模型
class NaiveBayesGaussian():
    def __init__(self, X_train, y_train):      # 这里X_train, y_train类型为np.array
        self.X_train = X_train
        self.y_train = y_train
        self.y_types = list(set(y))                  # y的所有可能取值

        def probability(X_train, y_train):
            '''
            :return: p_y: 字典。值: y的取值; 键: y的先验概率
                     pdfFunc_x_y: 字典。值: y的取值; 键: 所有特征X在y的条件下的条件概率密度函数构成的列表
            '''
            p_y = {}
            pdfFunc_x_y = {}
            for ck in self.y_types:
                p_y[ck] = sum(y_train == ck) / len(y_train)
                pdfFunc_x_y[ck] = GaussianFunc().pdfFunc(X_train[y_train == ck])
            return p_y, pdfFunc_x_y

        self.p_y, self.pdfFunc_x_y = probability(X_train, y_train)       # 防止每次预测都要调用一次


    def predict_single(self, x):
        res = []
        for i in range(len(self.y_types)):
            ck = self.y_types[i]
            p = self.p_y[ck]
            for j in range(len(self.pdfFunc_x_y[ck])):
                func = self.pdfFunc_x_y[ck][j]
                p = p * func(x[j])
            res.append(p)
        return self.y_types[res.index(max(res))]


    def predict(self, X_test):
        return [self.predict_single(x) for x in X_test]


    def accuracy(self, X_test, y_test):
        return np.mean(y_test == self.predict(X_test))


if __name__ == '__main__':
    X = load_iris().data
    y = load_iris().target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    fit = NaiveBayesGaussian(X_train, y_train)
    accuracy = fit.accuracy(X_test, y_test)
    print('\t')
    print('the accuracy of predicting iris by NaiveBayesGaussian: ', accuracy)

    # 用sklearn里的多项式朴素贝叶斯模型GaussianNB，该模型用于各个特征服从正态分布的情况
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    accuracy1 = clf.score(X_test, y_test)
    print('the accuracy of predicting iris by GaussianNB in sklearn: ', accuracy1)