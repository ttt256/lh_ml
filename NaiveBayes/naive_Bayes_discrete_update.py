import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# 多分类的朴素贝叶斯模型（特征均为离散型变量）
class NaiveBayesDiscrete():
    def __init__(self, X_train, y_train, lam = 1):
        '''
        注意这里输入的X_train, y_train的类型很重要
        如果X_train本来就是dataframe，不加copy()，data和X_train就是同一个，再给data加y列，X_train也会改变
        y_train的类型要化为np.array才能加到data中，因为np.array和dataframe的索引不一样
        '''
        self.data = pd.DataFrame(X_train).copy()       # 训练集特征, np.array, (n, m), n个样本, m个特征, 转化为DataFrame
        self.data['y'] = np.array(y_train)             # 训练集标签, np.array, (n, ), K个类, 与X_train整合在一起，方便后面的groupby
        self.n, self.m = X_train.shape                 # 样本个数，特征个数
        self.K = len(np.unique(y_train))               # 种类数，np.unique()对np.array去重

        def probability():
            '''
            计算y的概率和x在y的条件下的概率。
            如果lambda = 0，就是极大似然估计；如果lambda != 0，就是贝叶斯估计。
            :return: p_y: y的概率。
                     p_x_y: x在y的条件下的概率，字典。
                            键: j,第j个特征；值: 第j个特征xj在y的条件下的概率，每个值都是一个dataframe。
            '''
            # 求y的先验概率p_y
            counts_y = self.data.groupby([self.data.iloc[:, -1]]).size()      # counts_y里储存的是(y=b)的个数, b为y可能的取值
            p_y = (counts_y + lam) / (self.n + self.K * lam)

            # 求x在y的条件下的概率p_x_y
            p_x_y = {}                      # 字典
            for j in range(self.m):         # 第j个特征
                # 有些分组，比如(xj=1, y=0)不存在，先unstack()，再将unstack()中的缺失值NA替换为0
                # counts_x_y里储存的是(xj=a, y=b)的个数, (a,b)为xj,y可能的取值
                counts_x_y = self.data.groupby([self.data.iloc[:, j], self.data.iloc[:, -1]]).size().unstack().fillna(0)
                p_x_y[j] = (counts_x_y + lam) / (counts_x_y.sum() + len(counts_x_y) * lam)       # len(counts_x_y)为第j个特征xj可能取值的个数

            return p_y, p_x_y

        self.p_y, self.p_x_y = probability()


    def predict_single(self, x):
        '''
        预测单个目标值。
        :param x: 要预测的目标点
        :return: 预测值
        '''
        res = []        # 保存y的后验概率
        for k in range(self.K):             # y的第k个取值
            ck = self.p_y.index[k]          # self.p_y.index为y的所有可能取值，类型为np.array
            p = self.p_y[ck]                # self.p_y[ck]为y=ck的先验概率
            for j in range(self.m):         # 第j个特征xj
                # self.p_x_y[j]为第j个特征xj关于各个y的条件概率，.loc[x[j], ck]为x=x[j]在y=ck的条件下的概率
                p = p * self.p_x_y[j].loc[x[j], ck]
            res.append(p)
        return self.p_y.index[res.index(max(res))]        # res.index(max(res))后验概率最大的y的索引


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
    print('从头实现的Naive Bayes:')
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
    print('the predict_y of x in book example4.1 by NaiveBayesDiscrete: ', y_pred)


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
    print('the accuracy of predicting iris by NaiveBayesDiscrete: ', accuracy2)


    # 3. 用mushrooms数据集测试
    # mushrooms数据集共有22个特征，8124个样本，该数据集的每个特征都是离散型的
    df = pd.read_table('mushrooms.txt', sep = ',')
    df.head()
    X_train3, X_test3, y_train3, y_test3 = train_test_split(df.iloc[:, 1: ], df['class'], test_size=0.2, random_state=0)
    fit3 = NaiveBayesDiscrete(X_train3, y_train3)
    accuracy3 = fit3.accuracy(X_test3, y_test3)
    print('the accuracy of predicting mushrooms dataset by NaiveBayesDiscrete: ', accuracy3)




    # 用sklearn里的多项式朴素贝叶斯模型MultinomialNB，该模型用于各个特征服从多项式分布的情况
    # sklearn中的贝叶斯估计需要将数据数值化，例将x='S','M','L'化为x=0,1,2
    print('\t')
    from sklearn.naive_bayes import MultinomialNB

    # 创建实例
    clf = MultinomialNB()

    print('Numerical Features and Labels:')
    # 1. 用例4.1的数据测试
    # 数据预处理，将特征数值化
    X_train1_num = pd.DataFrame(X_train1)      # 利用dataframe的map
    X_train1_num.iloc[:, 1] = X_train1_num.iloc[:, 1].map({'S': 0, 'M': 1, 'L': 2})
    # 训练模型
    clf.fit(X_train1_num, y_train1)
    # 预测数据
    x_num = np.array([2, 0])
    y_predict = clf.predict(x_num.reshape((1, 2)))
    print('the predict_y of x in book example4.1 by MultinomialNB in sklearn: ', y_predict[0])


    # 2. 用iris数据集测试
    clf.fit(X_train2, y_train2)
    y_predict = clf.predict(X_test2)
    # 计算精确度
    acc2 = clf.score(X_test2, y_test2)
    print('the accuracy of predicting iris by MultinomialNB in sklearn: ', acc2)


    # 3. 用mushrooms数据集测试
    # 数据预处理，mushrooms数据集的特征全为str，标签也为str，所以定义函数来将特征、标签数值化

    # 特征数值化
    X_train3_num, X_test3_num = X_train3.copy(), X_test3.copy()      # 原数据集下面还要用
    y_train3_num, y_test3_num = y_train3.copy(), y_test3.copy()
    features = map(list, zip(*np.array(X_train3_num)))              # map的每一个位置是所有样本的每个特征的值
    features = [sorted(list(set(feature))) for feature in features]
    # 列表的每一个位置是每个特征的所有可能取值类型,sorted()保证每次顺序一致，不然每次结果都不一样
    # 不能用list.sort(),sort()返回None
    features_dics = [{val: i for i, val in enumerate(feature)} for feature in features]
    # 列表的每一个位置是一个字典，（键，值）：（每个特征的可能取值，取值的索引（这个取值是这个特征的第几个取值））
    for i in range(X_train3_num.shape[1]):         # 对每个特征
        X_train3_num.iloc[:, i] = X_train3_num.iloc[:, i].map(features_dics[i])

    # 标签数值化
    label_dic = {val: i for i, val in enumerate(sorted(list(set(y_train3_num))))}
    y_train3_num = y_train3_num.map(label_dic)

    # 将测试集特征、标签也数值化
    for i in range(X_train3_num.shape[1]):
        X_test3_num.iloc[:, i] = X_test3_num.iloc[:, i].map(features_dics[i])
    y_test3_num = y_test3_num.map(label_dic)

    clf.fit(X_train3_num, y_train3_num)
    acc3 = clf.score(X_test3_num, y_test3_num)
    print('the accuracy of predicting mushrooms dataset by MultinomialNB in sklearn: ', acc3)
    # sklearn中的MultinomialNB连例4.1的结果算出来都不一样，这是因为这些数据不能直接使用Multinomial这个模型




    # 使用OneHotEncoder
    print('\t')
    print('Use OneHotEncoder:')
    from sklearn.preprocessing import OneHotEncoder

    enc = OneHotEncoder()

    # 1. 用例4.1的数据测试
    X_train1 = enc.fit_transform(X_train1).toarray()
    x = enc.transform(x.reshape((1, 2))).toarray()

    clf.fit(X_train1, y_train1)
    y_predict = clf.predict(x)
    print('the predict_y of x in book example4.1 by MultinomialNB in sklearn: ', y_predict[0])

    # 2. 用iris数据集测试
    X_train2 = enc.fit_transform(X_train2)
    X_test2 = enc.transform(X_test2)

    clf.fit(X_train2, y_train2)
    acc2 = clf.score(X_test2, y_test2)
    print('the accuracy of predicting iris by MultinomialNB in sklearn: ', acc2)

    # 3. 用mushrooms数据集测试
    enc.fit(X_train3)
    X_train3 = enc.transform(X_train3).toarray()
    X_test3 = enc.transform(X_test3).toarray()

    clf.fit(X_train3, y_train3)
    acc3 = clf.score(X_test3, y_test3)
    print('the accuracy of predicting mushrooms dataset by MultinomialNB in sklearn: ', acc3)
