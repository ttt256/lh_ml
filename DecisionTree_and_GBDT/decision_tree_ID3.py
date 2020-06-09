import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split


# 定义节点类，树
class Tree:
    def __init__(self, node_type, label=None, feature_name=None):
        """
        叶结点(leaf)的属性有label, 其余为None;
        内部结点(internal)的属性有feature_name, 其余为None。
        """
        self.node_type = node_type          # 叶结点(leaf)或内部结点(internal)
        self.label = label
        self.feature_name = feature_name
        self.child = {}                     # 用字典来存孩子结点，字典的键为特征，值为子树Tree类

    def add_tree(self, feature, tree):
        self.child[feature] = tree


class DecisionTree:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.tree = None

    # 计算信息增益
    @staticmethod
    # 静态方法，对类和实例都可以调用该函数
    def entropy(y):
        """
        计算经验熵
        :param y: dataframe, 数据集的标签
        :return: 熵
        """
        p_y = y.value_counts() / len(y)
        ent = sum(p_y.map(lambda x: - x * math.log(x, 2)))
        return ent

    @staticmethod
    def cond_entropy(X, y, axis):
        """
        计算经验条件熵
        :param X: dataframe, 数据集的特征
        :param y: dataframe, 数据集的标签
        :param axis: 特征的维度
        :return: 条件熵
        """
        data = pd.concat([X, y], axis=1)     # 合并X，y
        p_x = data.groupby(data.iloc[:, axis]).size() / len(data)
        size = data.groupby([data.iloc[:, -1], data.iloc[:, axis]]).size().unstack()
        p_y_x = size / size.sum()
        cond_ent = sum(p_x * p_y_x.applymap(lambda x: - x * math.log(x, 2)).sum())
        return cond_ent

    @ staticmethod
    def info_gain(X, y, axis):
        return DecisionTree.entropy(y) - DecisionTree.cond_entropy(X, y, axis)

    def train(self, X_train, y_train):
        # 步骤(1)：所有实例属于同一类
        label_set = set(y_train)
        if len(label_set) == 1:
            return Tree('leaf', label_set.pop())               # 集合没有索引
        # 步骤(2)：特征为空
        if not X_train.shape[1]:
            return Tree('leaf', y_train.value_counts().idxmax())     # 实例数最大的类
        # 步骤(3)：计算信息增益，并选择信息增益最大的特征
        max_gda, max_feature_name = 0, None
        n = X_train.shape[1]           # 特征个数
        for axis in range(n):
            gda = DecisionTree.info_gain(X_train, y_train, axis)
            if gda > max_gda:
                max_gda = gda
                max_feature_name = X_train.columns[axis]
        # 步骤(4)：小于阈值
        if max_gda < self.epsilon:
            return Tree('leaf', y_train.value_counts().idxmax())
        # 步骤(5)：递归
        tree = Tree('internal', feature_name=max_feature_name)
        features = set(X_train[max_feature_name].values)
        for feature in features:
            index = X_train[max_feature_name] == feature
            sub_X = X_train[index]
            del sub_X[max_feature_name]
            sub_y = y_train[index]
            tree.add_tree(feature, self.train(sub_X, sub_y))
        return tree

    def fit(self, X_train, y_train):
        self.tree = self.train(X_train, y_train)

    def predict(self, X_test):
        y_pred = []
        for i in range(X_test.shape[0]):
            tree = self.tree
            while tree.node_type != 'leaf':
                feature_name = tree.feature_name
                feature = X_test.iloc[i][feature_name]
                tree = tree.child[feature]
            y_pred.append(tree.label)
        return y_pred

    def score(self, X_test, y_test):
        return np.mean(y_test == self.predict(X_test))



if __name__ == '__main__':
    # 1. 用书上的例5.1测试
    datasets = [['青年', '否', '否', '一般', '否'],
                ['青年', '否', '否', '好', '否'],
                ['青年', '是', '否', '好', '是'],
                ['青年', '是', '是', '一般', '是'],
                ['青年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '好', '否'],
                ['中年', '是', '是', '好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '好', '是'],
                ['老年', '是', '否', '好', '是'],
                ['老年', '是', '否', '非常好', '是'],
                ['老年', '否', '否', '一般', '否'],
                ]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    data = pd.DataFrame(datasets, columns=labels)
    # 这里的数据集很小，其实划分没有什么意义
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, : -1], data['类别'], test_size=0.2)
    clf = DecisionTree()
    clf.fit(X_train, y_train)
    clf.predict(X_test)
    clf.score(X_test, y_test)

    # 2. 用mushrooms数据集测试
    # mushrooms数据集共有22个特征，8124个样本，该数据集的每个特征都是离散型的
    df = pd.read_table('mushrooms.txt', sep=',')
    df.head()
    X_train2, X_test2, y_train2, y_test2 = train_test_split(df.iloc[:, 1:], df['class'], test_size=0.2, random_state=15)
    clf = DecisionTree()
    clf.fit(X_train2, y_train2)
    accuracy = clf.score(X_test2, y_test2)
    print('the accuracy of predict predicting mushrooms dataset by DecisionTree: ', accuracy)


    # 用sklearn里的决策树实现：分类树 DecisionTreeClassifier，回归树 DecisionTreeRegressor

    # 将数据转为独热编码的形式
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder()
    enc.fit(X_train2)
    X_train2 = enc.transform(X_train2).toarray()
    X_test2 = enc.transform(X_test2).toarray()
    enc.fit(pd.DataFrame(y_train2))             # series是一维的，这里的参数要求两维，所以化为Dataframe
    y_train2 = enc.transform(pd.DataFrame(y_train2)).toarray()
    y_test2 = enc.transform(pd.DataFrame(y_test2)).toarray()

    from sklearn.tree import DecisionTreeClassifier
    clf1 = DecisionTreeClassifier()
    clf1.fit(X_train2, y_train2)
    accuracy1 = clf1.score(X_test2, y_test2)
    print('the accuracy predicting mushrooms dataset by DecisionTreeClassifier in sklearn: ', accuracy1)