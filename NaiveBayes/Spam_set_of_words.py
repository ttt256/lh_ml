import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from naive_Bayes_discrete_update import NaiveBayesDiscrete

# 导入数据
df = pd.read_table('SMSSpamCollection.txt', names = ['label', 'message'])
df.head()

# 将数据标签化为二元变量
df['label'] = df['label'].map({'ham': 0, 'spam': 1})     # 0不是垃圾文件，1是垃圾文件
df.head()

# 从头实现词集模型set of words（词出现为1，没出现为0）
# 将message中的所有字符转换为小写
df['message'] = df.message.map(lambda x: x.lower())
# 去标点
df['message'] = df.message.str.replace('[^\w\s]', '')
# 将message化为单个单词, 有些分词后的message中含有空字符串''，所以要使用filter，filter返回迭代器，所以要加list
df['message'] = df.message.map(lambda x: list(filter(None, x.split(' '))))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=0)

# 考虑到这个数据集规模不大，并且处理的是短信，并不是电子邮件这样的更庞大文本来源，所以这里不设置去停用词
def createVocabList(dataset):
    '''记录训练集中的所有词，创建词表'''
    vocabSet = set()               # 记录训练集中的所有词
    for message in dataset:
        vocabSet = vocabSet | set(message)    # 求并集
    return list(vocabSet)                     # 这里化为list是因为后面需要用到索引，而集合没有索引

def setOfWords(vocabList, dataset):
    '''
    创建词集模型的词向量。
    :param dataset: pd.DataFrame
    :return: pd.DataFrame
    '''
    wordArray = np.zeros((len(dataset), len(vocabList)))
    for i in range(len(dataset)):
        for word in dataset.iloc[i]:       # dataset为pd.dataframe
            if word in vocabList:
                wordArray[i, vocabList.index(word)] = 1
    return pd.DataFrame(wordArray)

vocabList = createVocabList(X_train)
X_train = setOfWords(vocabList, X_train)
X_test = setOfWords(vocabList, X_test)

if __name__ == '__main__':
    print('\t')
    print('从头实现的 set of words:')
    # 调用上个文件中实现的多分类的贝叶斯模型
    fit = NaiveBayesDiscrete(X_train, y_train)         # 注意这里要X_train, y_train的类型为pd.DataFrame
    accuracy = fit.accuracy(X_test, y_test)
    print('the accuracy of spam collection by set of words: ', accuracy)

