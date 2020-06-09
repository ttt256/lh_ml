import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# 定义CART回归树类
class CARTNode:
    def __init__(self, node_type, feature_name=None, feature=None, y=None):
        """
        叶结点(leaf)的属性有y, 其余为None;
        内部结点(internal)的属性有feature_name和feature, 其余为None。
        """
        self.node_type = node_type     # 叶结点(leaf)或内部结点(internal)
        self.feature_name = feature_name
        self.feature = feature
        self.y = y
        self.left = None
        self.right = None


class GBDTRegression:
    def __init__(self):
        self.tree = None

    @staticmethod
    def find_split_point(X_train, y_train):
        """
        按平方误差最小化原则找到最佳切分点
        :param X_train: 训练集特征
        :param y_train: 训练集y值
        :return: 切分特征与切分点
        """
        min_loss = float('inf')
        best_feature_name = -1
        best_feature = -1
        X_left, X_right = None, None
        y_left, y_right = None, None

        n = X_train.shape[1]
        for axis in range(n):
            column = X_train.iloc[:, axis]
            category = sorted(set(column))      # sorted之后又变为list了
            # 若类型小于等于10，认为是离散数据，进行精确切分
            # 若类型大于10，认为是连续变量，进行10分位点切分
            if len(category) <= 10:
                split_point = category
            else:
                # 使用np.arrange来每次找到1/10数据点所在的索引
                # 然后进行切分
                split_point = np.arange(0, len(category), len(category) // 10)
                split_point = [category[split_point[i]] for i in range(len(split_point))]

            for point in split_point:
                left = y_train[column <= point]
                right = y_train[column > point]

                c_left = np.mean(left)
                c_right = np.mean(right)

                loss = np.sum((left - c_left) ** 2) + np.sum((right - c_right) ** 2)
                if loss < min_loss:
                    min_loss = loss
                    best_feature_name = X_train.columns[axis]
                    best_feature = point
                    X_left, X_right = X_train[column <= point], X_train[column > point]
                    y_left, y_right = y_train[column <= point], y_train[column > point]
        return best_feature_name, best_feature, X_left, X_right, y_left, y_right

    def createCART(self, X_train, y_train, deep, max_depth):
        """
        创建CART回归树
        :param X_train: 训练集特征
        :param y_train: 训练集y值
        :param max_depth: 回归树最大深度
        :return: tree
        """
        if deep < max_depth and len(X_train) >= 2:
            best_feature_name, best_feature, X_left, X_right, y_left, y_right = \
            self.find_split_point(X_train, y_train)
            tree = CARTNode('interval', feature_name=best_feature_name, feature=best_feature)
            tree.left = self.createCART(X_left, y_left, deep + 1, max_depth)
            tree.right = self.createCART(X_right, y_right, deep + 1, max_depth)
            return tree
        else:
            return CARTNode('leaf', y=np.mean(y_train))

    @staticmethod
    def predict_for_rm(x, tree, alpha):
        """
        获得前一轮即第m-1棵树对单个样本的预测值，从而获得残差
        :param x: 单个样本特征
        :param tree: 第m-1棵树
        :param alpha: 正则化系数
        :return: 第m-1棵树预测的值
        """
        while tree.node_type != 'leaf':
            feature_name = tree.feature_name
            feature = tree.feature
            if x[feature_name] <= feature:
                tree = tree.left
            else:
                tree = tree.right
        return alpha * tree.y

    def gradientBoosting(self, X_train, y_train, alpha, epoch, max_depth=4):
        """
        :param X_train: 训练集特征
        :param y_train: 训练集y值
        :param alpha: 正则化系数，防止过拟合
        :param epoch: 迭代次数
        :param max_depth: 树的最大深度
        :return:
        """
        tree_list = []      # 用来存储所有树
        # 第一步，初始化fx0，即找到使得损失函数最小的c
        # 没有切分特征，所有值均预测为样本点均值
        fx0 = CARTNode('leaf', y=np.mean(y_train))
        tree_list.append(fx0)

        # 开始迭代训练，对每一轮的残差拟合回归树
        for i in range(epoch - 1):
            # 更新样本y值，rmi=yi-fmx
            if i == 1:
                y_train = y_train - fx0.y
            else:
                for i in range(len(X_train)):
                    y_train.iloc[i] = y_train.iloc[i] - self.predict_for_rm(X_train.iloc[i], tree_list[-1], alpha)
            # 上面已经将样本值变为了残差，下面对残差拟合一颗回归树
            fx = self.createCART(X_train, y_train, deep=0, max_depth=max_depth)
            tree_list.append(fx)
        self.tree = tree_list

    def predict(self, X_test, alpha):
        """
        对单个样本进行预测
        :param X_test: 单个样本特征
        :param tree_list: 所有树的列表
        :param alpha: 正则化系数
        :return: 预测值
        """
        y_pred = []

        tree_list = self.tree
        m = len(tree_list)
        for j in range(len(X_test)):
            y = 0
            for i in range(m):
                tree = tree_list[i]
                if i == 0:
                    y += tree.y
                else:
                    y += self.predict_for_rm(X_test.iloc[j], tree, alpha)
            y_pred.append(y)
        return y_pred

    def score(self, X_test, y_test, alpha):
        return np.mean((y_test - self.predict(X_test, alpha)) ** 2)


def load_data(filename):
    data = pd.read_csv(filename)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    print('Load data...')
    filename = 'boston_house_prices.csv'
    X_train, X_test, y_train, y_test = load_data(filename)

    gbdt = GBDTRegression()
    gbdt.gradientBoosting(X_train, y_train, alpha=0.12, epoch=11, max_depth=4)

    mse = gbdt.score(X_test, y_test, alpha=0.12)

    y_pred = gbdt.predict(X_test, alpha=0.12)
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)

    print('GBDT performance on boston_house_price dataset:')
    print('================================================')
    print("%s %s" % ("mse".center(10), "r2_score".center(10)))
    print("%s %s" % (
                        ("%.4f" % mse).center(10),
                        ("%.4f" % r2).center(10)))

    """
    GBDT performance on boston_house_price dataset:
    ================================================
       mse      r2_score 
      9.4584     0.8186  
    """