import numpy as np
import pandas as pd
import random


def normalization(X):
    """
    对每个 feature 标准化。
    :param X: features
    :return: 标准化后的 X
    """
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)      # 防止分母为0
    return X


def train_val_split(X, Y, val_ratio=0.25, random_seed=0):
    """
    划分训练集和交叉验证集。
    :param X: 数据集 features
    :param Y: 数据集 target
    :param val_ratio: 划分交叉验证集占的比例
    :param random_seed: 随机种子
    :return: X_train, Y_train, X_test, Y_test
    """
    n = len(X)
    random.seed(random_seed)
    index = np.arange(n)
    random.shuffle(index)
    train_size = int(n * (1 - val_ratio))
    return X[index[: train_size]], Y[index[: train_size]], X[index[train_size: ]], Y[index[train_size: ]]


class LRModel:
    def __init__(self):
        self.w = None     # 含bias项

    @staticmethod
    def sigmoid(z):
        # 防止溢出
        if z >= 0:
            return np.clip(1 / (1 + np.exp(-z)), 1e-8, 1 - 1e-8)
        else:
            return np.clip(np.exp(z) / (1 + np.exp(z)), 1e-8, 1 - 1e-8)

    def f(self, X):
        """
        LR function.
        :param X: features
        :param w: weight向量，np.array, (n, m), m为特征个数
        :param b: bias
        :return: 预测的Y在X的条件下的概率，np.array, (n, 1)
        """
        return np.array(list(map(LRModel.sigmoid, np.dot(X, self.w))))

    def cross_entropy_loss(self, P, Y, lam=0):
        """
        计算损失函数（LR的损失函数为交叉熵）。
        :param P: 预测的Y在X的条件下的概率
        :param Y: 真实的Y
        :return: 损失函数
        """
        return - sum(Y * np.log(P) + (1 - Y) * np.log(1 - P)) + lam * sum(self.w ** 2)

    def fit(self, X_train, Y_train, iter=11, learning_rate=0.03, batch_size=8, lam=0):
        """
        训练模型，这里用的是 mini-batch Gradient Descent。
        :param X_train: 训练集 features, np.array, (n, m)
        :param Y_train: 训练集 target, np.array, (n, 1)
        :param iter: 迭代次数
        :param batch_size: mini-batch 梯度下降每次的样本量
        :param lam: 正则化
        :return: None, 得到模型的 weights vector w 和 bias b
        """
        n, m = X_train.shape
        self.w = np.zeros((m, 1))
        # mini-batch
        step = 1      # 梯度下降次数
        for t in range(iter):
            # 打乱训练集顺序
            randomize = np.arange(n)
            np.random.shuffle(randomize)
            X_train = X_train[randomize]
            Y_train = Y_train[randomize]
            for i in range(n // batch_size):
                X = X_train[i * batch_size: (i + 1) * batch_size]
                Y = Y_train[i * batch_size: (i + 1) * batch_size]
                grad = np.dot(X.T, self.f(X) - Y) + 2 * lam * self.w
                self.w -= learning_rate * grad / np.sqrt(step)          # Adaptive Learning rate
                step += 1

    def predict(self, X):
        """
        :return: 预测的target
        """
        P = self.f(X)
        return np.round(P)

    def accuracy(self, X, Y):
        return np.mean(Y == self.predict(X))


if __name__ == '__main__':
    # 导入数据
    X = pd.read_csv('X_train').to_numpy()[:, 1:]  # 第一列是id
    Y = pd.read_csv('Y_train').to_numpy()[:, 1:]
    # normalization
    X = normalization(X)
    X1 = np.insert(X, 0, 1, axis=1)

    # 划分训练集与交叉验证集
    X_train, Y_train, X_validation, Y_validation = train_val_split(X1, Y, val_ratio=0.1, random_seed=0)

    clf = LRModel()

    '''
    # 选learning_rate
    for learning_rate in (0.01, 0.03, 0.1, 0.3, 1):
        clf.fit(X_train, Y_train, learning_rate=learning_rate)
        print(clf.accuracy(X_validation, Y_validation))
    # 选learning_rate对应的迭代次数
    for iter in (5, 7, 9, 11, 13):
        clf.fit(X_train, Y_train, iter=iter)
        print(clf.accuracy(X_validation, Y_validation))
    '''

    # k-fold cross validation
    from sklearn.model_selection import KFold
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    '''
    # 选lambda
    for lam in (0, 0.005, 0.01, 0.05, 0.1, 0.5, 1):
        loss_val = 0
        for train_index, val_index in kf.split(X1):
            clf.fit(X1[train_index], Y[train_index], lam=lam)
            loss_val += clf.cross_entropy_loss(clf.f(X1[val_index]), Y[val_index], lam=lam) / k
        print(lam, loss_val)
    '''

    # 将选出的learning_rate, iter, lambda代入模型，求对验证集的平均accuracy
    accuracy = 0
    for train_index, val_index in kf.split(X1):
        clf.fit(X1[train_index], Y[train_index])
        accuracy += clf.accuracy(X1[val_index], Y[val_index]) / k
    print('the mean of accuracy is:', accuracy)

    # 导入测试集数据，测试集数据预处理方式和训练集一样
    X_test = pd.read_csv('X_test').to_numpy()[:, 1:]
    X_test = normalization(X_test)
    X_test1 = np.insert(X_test, 0, 1, axis=1)
    predictions = clf.predict(X_test1)
    # save prediction to CSV file
    with open('output_logistic.csv', mode='w') as submit_file:
        submit_file.write('id,label\n')
        for i, label in enumerate(predictions):
            submit_file.write('{},{}\n'.format(i, int(label)))


    # 直接调用sklearn中的LogisticRegression
    from sklearn.linear_model import LogisticRegression
    clf1 = LogisticRegression()
    clf1.fit(X, Y.flatten())
    predictions = clf1.predict(X_test)
    # save prediction to CSV file
    with open('output_logistic1.csv', mode='w') as submit_file:
        submit_file.write('id,label\n')
        for i, label in enumerate(predictions):
            submit_file.write('{},{}\n'.format(i, int(label)))
