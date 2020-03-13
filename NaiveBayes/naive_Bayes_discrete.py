import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# 多分类的朴素贝叶斯模型（特征均为离散型变量）
class NaiveBayesDiscrete():
    def __init__(self, X_train, y_train, lam = 1):
        self.X_train = pd.DataFrame(X_train)      # 训练集特征, np.array, (n, m), n个样本, m个特征, 转化为DataFrame
        self.y_train = pd.DataFrame(y_train)      # 训练集标签, np.array, (n, ), K个类, 转化为DataFrame
        self.n, self.m = X_train.shape            # 样本个数，特征个数
        self.K = len(np.unique(y_train))          # 种类数，np.unique()对np.array去重

        def probability():
            '''
            计算y的概率和x在y的条件下的概率。
            如果lambda = 0，就是极大似然估计；如果lambda != 0，就是贝叶斯估计。
            y_counts: 记录了y的所有类型以及对应的个数
            y_counts.index: 记录了y的所有类型
            y_counts.values: 记录了y的每个类型对应的个数
            X_counts: 记录了X第j个特征的所有类型以及对应的个数
            X_counts.index: 记录了X第j个特征的所有类型
            X_counts.values: 记录了X第j个特征的每个类型对应的个数
            '''
            y_counts = self.y_train.iloc[:, 0].value_counts()            # Y=ck的个数，value_counts()是对Series使用的，所以这里要取.iloc[0]
            p_y = (y_counts + lam) / (self.n + self.K * lam)             # Y=ck的概率
            p_x_y = {}                                  # 键(j, i, k): (第j个特征, 第j个特征的第i个取值, y的第k个取值); 值P(Xj=aji|Y=ck)
            for j in range(self.m):                     # 第j个特征
                for k in range(self.K):                 # y的第k个取值
                    X_counts = self.X_train[(self.y_train == y_counts.index[k]).values].iloc[:, j].value_counts()     # y == k为dataframe，要用.values化为np.array
                    S = len(X_counts.index)             # 第j个特征可取的值有S个
                    for i in range(S):
                        p_x_y[(j, X_counts.index[i], y_counts.index[k])] = \
                            ((X_counts.values[i] + lam) / (y_counts.values[k] + S * lam))           # 这里i的类型为str，因为X1的取值为'S','M','L'
            return y_counts, p_y, p_x_y

        self.y_counts, self.p_y, self.p_x_y = probability()


    def predict_single(self, x):
        '''
        预测单个目标值。
        :param x: 要预测的目标点
        :return: 预测值
        '''
        res = []        # 保存y的后验概率
        for k in range(self.K):
            p = self.p_y.iloc[k]
            for j in range(self.m):
                if (j, x[j], self.y_counts.index[k]) not in self.p_x_y:
                    p = 0
                    break
                    """
                    假如训练集中(x=1, y=0)的个数为0，那么在p_x_y中，P(x=1|y=0)并没有计算，
                    且在P(x|y=0)的计算中，x的种类S少记了一种。
                    这里取P(x=1|y=0)=0是合理的（后面测试的正确率也很高），但是好像并不准确。
                    """
                p *= self.p_x_y[(j, x[j], self.y_counts.index[k])]
            res.append(p)
        return self.y_counts.index[res.index(max(res))]        # res.index(max(res))后验概率最大的y的索引


    def predict(self, X_test):
        '''
        预测多个目标值。
        :param X_test: 测试集, 注意这里的测试集是np.array类型
        :return: 测试集的预测值
        '''
        X_test = np.array(X_test)
        return [self.predict_single(x) for x in X_test]


    def accuracy(self, X_test, y_test):
        '''模型精确度'''
        return np.mean(y_test == self.predict(X_test))



if __name__ == '__main__':
    print('\t')
    # 1. 用例4.1的数据测试
    X_train1 = np.array([
        [1, "S"],
        [1, "M"],
        [1, "M"],
        [1, "S"],
        [1, "S"],
        [2, "S"],
        [2, "M"],
        [2, "M"],
        [2, "L"],
        [2, "L"],
        [3, "L"],
        [3, "M"],
        [3, "M"],
        [3, "L"],
        [3, "L"]
    ])
    y_train1 = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    x = np.array([2, 'S'])
    fit1 = NaiveBayesDiscrete(X_train1, y_train1, lam=1)
    # 预测单个点
    y_pred = fit1.predict_single(x)
    print('the predict_y of x in book example4.1: ', y_pred)

    # 2. 用iris数据集测试
    iris = load_iris()
    X = np.array(iris.data)
    y = np.array(iris.target)
    # iris的四个特征都为连续变量，所以要先对它们进行离散化，利用pandas中的cut()
    # 这里用等宽法（同一特征的每个分组距离一样）
    bins0 = [4, 5, 6, 7, 8]
    bins1 = [2, 3, 4, 5]
    bins2 = [1, 3, 5, 7]
    bins3 = [0, 1, 2, 3]
    X0 = pd.cut(X[:, 0], bins0, labels = [0, 1, 2, 3], right = False)    # right = False表示左闭右开
    X1 = pd.cut(X[:, 1], bins1, labels = [1, 2, 3], right = False)       # 因为能取到2所以用左闭右开
    X2 = pd.cut(X[:, 2], bins2, labels = [0, 1, 2], right = False)
    X3 = pd.cut(X[:, 3], bins3, labels = [0, 1, 2], right = False)
    new_X = np.array([X1, X2, X3, X3]).transpose()      # 转置

    X_train2, X_test2, y_train2, y_test2 = train_test_split(new_X, y, test_size=0.2, random_state=1)
    fit2 = NaiveBayesDiscrete(X_train2, y_train2, lam=1)
    accuracy2 = fit2.accuracy(X_test2, y_test2)
    print('the accuracy of predicting iris dataset by Naive Bayes: ', accuracy2)


    # 3. 用mushrooms数据集测试
    # mushrooms数据集共有22个特征，8124个样本，该数据集的每个特征都是离散型的
    df = pd.read_table('mushrooms.txt', sep = ',')
    df.head()
    X_train3, X_test3, y_train3, y_test3 = train_test_split(df.iloc[:, 1: ], df['class'], test_size=0.2, random_state=1)
    fit3 = NaiveBayesDiscrete(X_train3, y_train3)
    accuracy3 = fit3.accuracy(X_test3, y_test3)
    print('the accuracy of predicting mushrooms dataset by Naive Bayes: ', accuracy3)




