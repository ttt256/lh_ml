import numpy as np
import pandas as pd
from hw2 import normalization, train_val_split


class ProbabilisticGenerative:
    def __init__(self):
        self.w = None
        self.b = None

    @staticmethod
    def sigmoid(z):
        if z >= 0:
            return np.clip(1 / (1 + np.exp(-z)), 1e-8, 1 - 1e-8)
        else:
            return np.clip(np.exp(z) / (1 + np.exp(z)), 1e-8, 1 - 1e-8)

    @staticmethod
    def mean_and_cov(X_train, Y_train):
        """求期望向量和方差阵"""
        index0 = (Y_train == 0).flatten()
        index1 = (Y_train == 1).flatten()
        X_train0 = X_train[index0]
        X_train1 = X_train[index1]
        mu0, mu1 = X_train0.mean(0), X_train1.mean(0)
        sigma0 = np.cov(X_train0, rowvar=False, bias=True)
        sigma1 = np.cov(X_train1, rowvar=False, bias=True)
        # 同方差假定
        N0, N1 = len(X_train0), len(X_train1)
        sigma = (N0 * sigma0 + N1 * sigma1) / (N0 + N1)
        return mu0, mu1, sigma, N0, N1

    def fit(self, X_train, Y_train):
        """训练模型。probabilisitic generative model有可解析的最优解，因此不必使用验证集。"""
        mu0, mu1, sigma, N0, N1 = ProbabilisticGenerative.mean_and_cov(X_train, Y_train)
        # Compute inverse of covariance matrix.
        # Since covariance matrix may be nearly singular, np.linalg.inv() may give a large numerical error.
        # Via SVD decomposition, one can get matrix inverse efficiently and accurately.
        U, S, V = np.linalg.svd(sigma, full_matrices=False)
        inv_sigma = np.matmul(V.T * 1 / S, U.T)
        self.w = np.dot((mu0 - mu1).T, inv_sigma)
        self.b = -0.5 * np.dot(mu0, np.dot(inv_sigma, mu0)) + 0.5 * np.dot(mu1, np.dot(inv_sigma, mu1)) + np.log(N0 / N1)

    def predict(self, X):
        # 注意：probabilistic generative model算出来的是P(Y=0|X)的概率，所以要用1减
        P = np.array(list(map(self.sigmoid, np.dot(X, self.w) + self.b)))
        return 1 - np.round(P)

    def accuracy(self, X, Y):
        return np.mean(Y == self.predict(X))


if __name__ == '__main__':
    # 数据处理方式与LR中一样，这里不用插入x0
    # 导入数据
    X_train = pd.read_csv('X_train').to_numpy()[:, 1:]  # 第一列是id
    Y_train = pd.read_csv('Y_train').to_numpy()[:, 1:]
    # normalization
    X_train = normalization(X_train)
    # 划分训练集与交叉验证集
    X_train, Y_train, X_validation, Y_validation = train_val_split(X_train, Y_train, val_ratio=0.1, random_seed=0)

    # 拟合模型
    clf = ProbabilisticGenerative()
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_train)
    accuracy = clf.accuracy(X_train, Y_train)
    print('the accuracy of probabilistic_generation is:', accuracy)    # 0.6781833138320165


