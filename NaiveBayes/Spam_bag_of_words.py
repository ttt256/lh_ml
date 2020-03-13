import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# 二分类的多项式型朴素贝叶斯模型
class Model():
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        def fit(X_train, y_train):
            '''
            计算Y=0和Y=1的先验概率；计算在Y=0和Y=1的条件下，每个单词出现的条件概率
            :param X_train: 训练集特征，np.array，(n, J)，每一行是一个词向量，记录在这个样本中这个单词出现的频数
            :param y_train: 训练集标签，np.array，(n, )，0不是垃圾文件，1是垃圾文件
            :return: p_0, p_1: Y=0和Y=1的先验概率
                     p_x_0, p_x_1: 在Y=0和Y=1的条件下，每个单词出现的条件概率
            '''
            p_1 = sum(y_train) / len(y_train)   # Y=1的先验概率
            p_0 = 1 - p_1                       # Y=0的先验概率
            n, J = X_train.shape                # 样本个数；特征个数，即词表长度
            x_0_num = np.ones(J)                # 在Y=0时，每个单词出现的次数（频数），从1开始是为了拉普拉斯平滑，防止出现为0的概率
            x_1_num = np.ones(J)                # 注意这里不能写x_0_num = x_1_num = np.ones(J)，不然x_0_num和x_1_num的地址一样！！！
            y_0_num = 2               # 记录Y=0的所有单词的个数（注意这里是单词个数不是样本个数），从2开始也是为了拉普拉斯平滑
            y_1_num = 2
            for i in range(n):
                if y_train[i] == 1:
                    x_1_num += X_train[i]           # 这里利用了numpy.array的直接相加，很方便
                    y_1_num += sum(X_train[i])      # 这里是加上第i个样本中的所有单词
                else:
                    x_0_num += X_train[i]
                    y_0_num += sum(X_train[i])
            p_x_0 = x_0_num / y_0_num           # 在Y=0的条件下，每个单词出现的条件概率
            p_x_1 = x_1_num / y_1_num           # 在Y=1的条件下，每个单词出现的条件概率
            return p_0, p_1, p_x_0, p_x_1

        self.p_0, self.p_1, self.p_x_0, self.p_x_1 = fit(X_train, y_train)     # 防止预测每一个目标点都要重新计算


    def predict_single(self, x):
        '''
        预测单个目标点。
        因为是求最大后验概率对应的y值，所以可以略掉系数，还可以取对数，从而避免下溢出或者浮点数舍入导致的错误。
        '''
        p0 = sum(x * np.log(self.p_x_0)) + np.log(self.p_0)
        p1 = sum(x * np.log(self.p_x_1)) + np.log(self.p_1)
        return 0 if p0 > p1 else 1


    def predict(self, X_test):
        return [self.predict_single(x) for x in X_test]


    def accuracy(self, X_test, y_test):
        return np.mean(y_test == self.predict(X_test))





# bag of words
# 1. 从头 bag of words
# 导入数据
df = pd.read_table('SMSSpamCollection.txt', names = ['label', 'message'])
df.head()
# 将数据标签化为二元变量
df['label'] = df['label'].map({'ham': 0, 'spam': 1})     # 0不是垃圾文件，1是垃圾文件
df.head()
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=0)
# 将message中的所有字符转换为小写
df['message'] = df.message.map(lambda x: x.lower())
# 去标点
df['message'] = df.message.str.replace('[^\w\s]', '')
# 将message化为单个单词, 有些分词后的message中含有空字符串''，所以要使用filter
df['message'] = df.message.map(lambda x: list(filter(None, x.split(' '))))
# 创建词表和词向量
def createVocabList(dataset):
    '''记录训练集中的所有词，创建词表'''
    vocabSet = set()
    for message in dataset:
        vocabSet = vocabSet | set(message)
    return list(vocabSet)

def bagOfWords(vocabList, dataset):
    '''
    创建词袋模型的词向量,
    :param dataset: pd.DataFrame
    :return: np.array
    '''
    wordArray = np.zeros((len(dataset), len(vocabList)))
    for i in range(len(dataset)):
        for word in dataset.iloc[i]:
            if word in vocabList:
                wordArray[i, vocabList.index(word)] += 1
    return wordArray

vocabList = createVocabList(X_train)
X_train = bagOfWords(vocabList, X_train)
X_test = bagOfWords(vocabList, X_test)
y_train, y_test = np.array(y_train), np.array(y_test)     # y要化为np.array

# 2. 直接调用 sklearn.feature_extraction.text 中的 CountVectorizer
# 导入数据
df = pd.read_table('SMSSpamCollection.txt', names = ['label', 'message'])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})     # 0不是垃圾文件，1是垃圾文件
X_train1, X_test1, y_train1, y_test1 = train_test_split(df['message'], df['label'], test_size=0.2, random_state=0)

count_vector = CountVectorizer()
X_train1 = count_vector.fit_transform(X_train1).toarray()      # 用训练集建立vocabulary表
X_test1 = count_vector.transform(X_test1).toarray()
y_train1, y_test1 = np.array(y_train1), np.array(y_test1)

if __name__ == '__main__':
    # 1. 从头实现的 bag of words
    print('\t')
    print('从头实现的 bag of words:')
    # 调用上面实现的二分类的多项式型朴素贝叶斯模型
    fit = Model(X_train, y_train)                    # 注意这里要X_train, y_train的类型为np.array
    accuracy = fit.accuracy(X_test, y_test)
    print('the accuracy of spam collection by bag of words: ', accuracy)

    # 用sklearn的MultinomialNB直接构造
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print('the accuracy of spam collection by bag of words and MultinomialNB in sklearn: ', acc)

    # 2. 调用 CountVectorizer 实现的 bag of words
    print('\t')
    print('调用 CountVectorizer 实现的 bag of words:')
    fit1 = Model(X_train1, y_train1)  # 注意这里要X_train, y_train的类型为np.array
    accuracy1 = fit1.accuracy(X_test1, y_test1)
    print('the accuracy of spam collection by bag of words: ', accuracy1)

    clf1 = MultinomialNB()
    clf1.fit(X_train1, y_train1)
    acc1 = clf1.score(X_test1, y_test1)
    print('the accuracy of spam collection by bag of words and MultinomialNB in sklearn: ', acc1)