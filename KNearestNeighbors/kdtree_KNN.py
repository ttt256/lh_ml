import numpy as np
import random
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


# 构造kd树
class KdNode():
    # 二叉树节点
    def __init__(self, val, label, left = None, right = None):
        self.val = val
        self.left = left
        self.right = right
        self.label = label

# 构造平衡kd树
class KdTree():
    def __init__(self, data, y):
        self.k = len(data[0])         # 特征数量
        self.y = y

        def createTree(data, y, j):
            '''
            构造平衡二叉树
            :param data: np.array, 训练数据集
            :param y: np.array, 训练数据集对应的标签
            :param j: 递归深度
            :return: 父节点
            '''
            # 递归停止条件
            if not len(data):
                return None
            # 递归过程
            y = y[data[:, j % self.k].argsort()]        # y按第j % self.k列排序，注意要先给y排序，再给data排序
            data = data[data[:, j % self.k].argsort()]  # data按第j % self.k列排序
            midIndex = len(data) // 2                   # 右中位数
            node = KdNode(data[midIndex], y[midIndex])
            node.left = createTree(data[: midIndex], y[: midIndex], j + 1)
            node.right = createTree(data[midIndex + 1:], y[midIndex + 1:], j + 1)
            return node

        self.root = createTree(data, y, 0)

'''
# 用书上例3.2检验构造kd树的程序是否正确
data = np.array([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]])
y = np.array([1,1,1,1,0,0])
kdtreeRoot = KdTree(data, y).root
# 前序遍历kd树，与图3.4对照
def preorder(node, res):
    if node:
        res.append(node.val)
    if node.left:
        preorder(node.left, res)
    if node.right:
        preorder(node.right, res)
    return res
res = preorder(kdtreeRoot, [])
res
'''

# 用kd树的KNN
class KdModel():
    def __init__(self, data, y):
        '''
        :param data: np.array, 构造kd树的训练数据集
        :param y: np.array, 训练数据集的标签
        :param x: 目标点
        '''
        self.data = data                         # 训练数据集
        self.y = y                               # 训练数据集的标签
        self.kdtreeRoot = KdTree(data, y).root   # 构造的kd树的根节点
        self.k = len(data[0])                    # 特征数量


    def distance(self, node, x):
        return sum((node.val - x) ** 2)    # 欧氏距离


    # 最近邻算法
    def nearest_neighbor(self, x):
        '''
        用kd树的最近邻搜索
        :return: 目标点x的最近邻点
        '''
        node = self.kdtreeRoot
        nearestNode = node
        nearestDistance = float('inf')

        def helper(node, j):
            nonlocal nearestNode, nearestDistance

            # 类似于二叉树的中序遍历
            # 递归停止条件
            if not node:
                return

            index = j % self.k
            # 因为kd树取的是右中位数，所以node.left为None时node.right一定为None
            # 遍历子节点
            if x[index] <= node.val[index]:
                helper(node.left, j + 1)
                brotherNode = node.right
            else:
                helper(node.right, j + 1)
                brotherNode = node.left

            # 遍历根节点
            if self.distance(node, x) < nearestDistance:
                nearestNode = node
                nearestDistance = self.distance(nearestNode, x)

            # 遍历另一个子节点
            if (node.val[index] - x[index]) ** 2 <= nearestDistance:
                helper(brotherNode, j + 1)

        helper(node, 0)
        return nearestNode


    def nearest_accuracy(self, X, y):
        '''
        :param X: 测试集, np.array, (n, self.k), n为测试集目标点个数, self.k为特征数量
        :return: np.array, (n, 1), 最近邻法预测的测试集的标签y_pred
        '''
        y_pred = [self.nearest_neighbor(x).label for x in X]
        return np.mean(y_pred == y)


    # k值近邻算法
    def k_nearest_neighbor(self, x, k):
        '''
        用kd树的k值近邻搜索，最近邻算法只需输入 k=1
        :return: 目标点x的k个最近邻点
        '''
        node = self.kdtreeRoot
        nearest = []                          # 用来保存k个最近邻的列表，（距离，点）,不能用字典，可能会有重复点
        knearestDistance = float('inf')       # 当前k个最近邻中的最远距离

        def helper(node, j):
            nonlocal nearest, knearestDistance, k

            # 递归停止条件
            if not node:
                return

            index = j % self.k

            # 遍历子节点
            if x[index] <= node.val[index]:
                helper(node.left, j + 1)
                brotherNode = node.right
            else:
                helper(node.right, j + 1)
                brotherNode = node.left

            # 遍历根节点
            if self.distance(node, x) < knearestDistance or len(nearest) < k:
                nearest.append((self.distance(node, x), node))
                if len(nearest) > k:
                    # index找出的是第一次出现的索引，有重复值也只会删去一个
                    max_index = nearest.index(max(nearest, key = lambda x: x[0]))
                    del nearest[max_index]
                knearestDistance = max(nearest, key = lambda x: x[0])[0]

            # 遍历另一个子节点
            if (x[index] - node.val[index]) ** 2 <= knearestDistance or len(nearest) < k:
                helper(brotherNode, j + 1)

        helper(node, 0)
        return [j[1] for j in nearest]


    def predict(self, X, k):
        '''
        :param X: 测试集, np.array, (n, self.k), n为测试集目标点个数, self.k为特征数量
        :param k: k值近邻法的k
        :return: np.array, (n, 1), 预测的测试集的标签y_pred
        '''
        knearestNode = [self.k_nearest_neighbor(x, k) for x in X]
        class_num = len(set(self.y))
        y_pred = []
        for nodes in knearestNode:
            class_count = [0 for _ in range(class_num)]           # 计数器
            for node in nodes:
                class_count[node.label] += 1
            y_pred.append(class_count.index(max(class_count)))
        return np.array(y_pred)

    def accuracy(self, X, y, k):
        '''
        :param X: 测试集, np.array, (n, self.k), n为测试集目标点个数, self.k为特征数量
        :param y: 测试集的标签y, np.array, (n, )
        :param k: k值近邻法的k
        :return: 准确率
        '''
        return np.mean(y == self.predict(X, k))




if __name__ == '__main__':
    # 加载数据集测试
    iris = load_iris()
    X = np.array(iris.data[: 100, :2])    # 为了方便可视化，只取两个特征和两个种类
    y = np.array(iris.target[: 100])

    # 选取80个作为训练集，20个作为测试集
    random.seed(0)
    index = random.sample(range(0, 100), 100)
    X_train, X_test = X[index[:80], :], X[index[80:], :]
    y_train, y_test = y[index[:80]], y[index[80:]]

    # 拟合模型
    fit = KdModel(X_train, y_train)
    # 预测
    # 最近邻算法
    acc = fit.nearest_accuracy(X_test, y_test)
    print('accuracy(k = 1):', acc)
    # k近邻法(k=3)
    y_pred = fit.predict(X_test, k = 3)
    accuracy = fit.accuracy(X_test, y_test, k = 3)
    print('accuracy(k = 3): ', accuracy)

    # 可视化
    # 数据分布
    plt.figure()
    plt.scatter(X[:50, 0], X[:50, 1], label = '0')
    plt.scatter(X[50:, 0], X[50:, 1], label = '1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()

    # 取一个点观察最近的k个邻点
    plt.figure()
    plt.scatter(X_train[:, 0], X_train[:, 1])
    i = 18      # i可以在0,19之间取值
    plt.scatter(X_test[i, 0], X_test[i, 1], color = 'yellow')      # 目标点
    points = fit.k_nearest_neighbor(X_test[i, :], k = 3)           # 目标点的最近k个点
    for p in points:
        plt.scatter(p.val[0], p.val[1],color = 'red')              # 有重复点，所以图上可能没显示出三个点
    plt.show()

    # 用iris的所有数据检验模型（4维数据）
    X1 = np.array(iris.data)
    y1 = np.array(iris.target)
    # 选120个数据作为训练集，30个数据作为测试集
    index1 = random.sample(range(0, 150), 150)
    X1_train, y1_train = X1[index1[: 120]], y1[index1[: 120]]
    X1_test, y1_test = X1[index1[120:]], y1[index1[120:]]
    fit1 = KdModel(X1_train, y1_train)
    accuracy1 = fit1.accuracy(X1_test, y1_test, k = 5)
    print('the accuracy of predicting iris by knn with kdtree:', accuracy1)



    # 用sklearn里的K近邻法模型
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
    df['target'] = iris.target
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(df[iris.feature_names], df['target'], test_size = 0.2, random_state = 0)
    # 创建实例（模型）
    clf = KNeighborsClassifier(n_neighbors = 5)      # 如果不选k值，函数会自动选择
    # 训练模型
    clf.fit(X_train, y_train)
    # 预测数据
    y_predict = clf.predict(X_test)
    # 计算精确度
    score = clf.score(X_test, y_test)
    print('the accuracy of predicting iris by knn in sklearn:', score)
