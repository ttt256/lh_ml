import numpy as np
import random
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

class Model():
    def perceptron_ori(self, X, y, eta = 1, epoch = 1000):
        '''
        感知机的原始形式
        :param X: 训练集, np.array, (m, n), n表示特征数量, m表示样本量
        :param y: np.array, (m, )
        :param eta: learning rate
        :param epoch: 连续epoch次迭代正确就停止训练
        :return: 模型参数w, b
        '''
        m, n = X.shape
        # 选取参数初值
        w, b = np.zeros((n, )), 0

        train_count = 0       # 记录迭代正确次数
        # 随机梯度下降法进行训练
        while True:
            train_count += 1
            if train_count > epoch:
                break
            # 随机选一个数据
            i = random.randint(0, m - 1)
            if y[i] * (np.dot(X[i], w) + b) <= 0:
                train_count = 0     # 重新记录次数
                w = w + eta * y[i] * X[i].T
                b = b + eta * y[i]
        return w, b

    def perceptron_dual(self, X, y, eta = 1, epoch = 1000):
        '''
        感知机的对偶形式
        '''
        m, n = X.shape
        # Gram矩阵
        G = np.dot(X, X.T)
        # 选取参数初值
        alpha, b = np.zeros((m, )), 0

        train_count = 0   # 记录迭代正确次数
        # 随机梯度下降
        while True:
            train_count += 1
            if train_count >= epoch:
                break
            i = random.randint(0, m - 1)
            if y[i] * (G[i, :].dot(alpha * y) + b) <= 0:
                train_count = 0
                alpha[i] = alpha[i] + eta
                b = b + eta * y[i]
        w = X.T.dot(alpha * y)
        return w, b

    def predict(self, w, b, X):
        '''
        :param w: 模型参数w
        :param b: 模型参数b
        :param X: 测试集, array, (m, n), n表示特征数量, m表示样本量
        :return: 预测的y值, array, (m, )
        '''
        return [1 if x.dot(w) + b > 0 else -1 for x in X]

    def accuracy(self, y_train, y_test):
        '''
        求模型的准确率
        :param y_test: 预测的y
        :param y: 真实的y
        :return: 准确率
        '''
        return np.mean(y_train == y_test)



if __name__ == '__main__':
    # 加载数据集
    iris = load_iris()
    # iris的前50个数据target为0，中间50个target为1，这里选取前100个数据
    # 这里只是为了检验模型的正确性，为了方便，所以只选两个特征
    X = np.array(iris.data[:100, :2])
    y = np.array([-1] * 50 + [1] * 50)

    # 选取80个作为训练集，20个作为测试集
    random.seed(0)
    index = random.sample(range(0, 100), 100)
    X_train, X_test = X[index[:80], :], X[index[80:], :]
    y_train, y_test = y[index[:80]], y[index[80:]]

    # 拟合模型
    fit1 = Model()
    w1, b1 = fit1.perceptron_ori(X_train, y_train)      # 如果用所有数据fit1.perceptron_ori(X, y)，因为所有数据线性可分，所以accuracy为1
    fit2 = Model()
    w2, b2 = fit2.perceptron_dual(X_train, y_train)
    print("w1, b1:", w1, b1)
    print("w2, b2:", w2, b2)

    # 预测
    y_pred = fit1.predict(w1, b1, X_test)
    acc1 = fit1.accuracy(y_pred, y_test)
    y_pred = fit2.predict(w2, b2, X_test)
    acc2 = fit2.accuracy(y_pred, y_test)
    print("accuracy1:", acc1)
    print("accuracy2:", acc2)

    # 可视化
    plt.figure()
    plt.scatter(X[:50,0], X[:50,1], label = '-1')
    plt.scatter(X[50:,0], X[50:,1], label = '1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    # 画出拟合直线
    x1 = np.linspace(4, 7, 100)
    x2 = -(b1 + w1[0] * x1) / w1[1]
    plt.plot(x1, x2, color = 'yellow', label = 'fit1')
    x1 = np.linspace(4, 7, 100)
    x2 = -(b2 + w2[0] * x1) / w2[1]
    plt.plot(x1, x2, color = 'red', label = 'fit2')
    plt.legend()
    plt.show()



    # 用sklearn里的感知机模型
    from sklearn.linear_model import Perceptron
    clf = Perceptron(random_state=100)
    clf.fit(X_train, y_train)
    # 权值矩阵
    print('w:', clf.coef_)
    # 偏置项
    print('b:', clf.intercept_)
    y_pred = clf.predict(X_test)
    acc = clf.score(X_test, y_test)
    print("accuracy:", acc)
    # 可视化
    plt.figure()
    plt.scatter(X[:50,0], X[:50,1], label = '-1')
    plt.scatter(X[50:,0], X[50:,1], label = '1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    # 画出拟合直线
    x1 = np.linspace(4, 7, 100)
    x2 = -(clf.intercept_ + clf.coef_[0][0] * x1) / clf.coef_[0][1]
    plt.plot(x1, x2, color = 'green')
    plt.show()