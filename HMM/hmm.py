"""
HMM典型模型是一个五元组：
1. 状态值集合
    (B, M, E, S): {B:begin, M:middle, E:end, S:single}
2. 观察值集合
    在ord中，中文编码大小为65536，总共4个状态，所以B矩阵4x65536
3. 转移概率矩阵A
4. 观察概率矩阵B
5. 初值状态概率向量pi

比如:
小明硕士毕业于中国科学院计算所

输出的状态序列为
BEBEBMEBEBMEBES

根据这个状态序列我们可以进行切词:
BE/BE/BME/BE/BME/BE/S

所以切词结果如下:
小明/硕士/毕业于/中国/科学院/计算/所

同时我们可以注意到：
B后面只可能接(M or E)，不可能接(B or S)。而M后面也只可能接(M or E)，不可能接(B, S)。
"""

import numpy as np


class HMM:
    def __init__(self):
        """
        HMM模型参数(A, B, pi)，
        A为状态转移矩阵，B为观测概率矩阵，pi为初始状态概率向量。"""
        self.A = None
        self.B = None
        self.pi = None

    def fit(self, filename):
        """
        根据语料库训练HMM模型，得到模型参数(A, B, pi)，
        :param filename: 语料库
        :return: A, B, pi
        """
        # 定义一个状态映射字典。方便我们定位状态在列表中对应位置
        status2num = {'B': 0, 'M': 1, 'E': 2, 'S': 3}

        A = np.zeros((4, 4))
        B = np.zeros((4, 65536))
        pi = np.zeros(4)

        # 读取语料库（语料库做好了切分，每个词已经用空格隔开了），按行读取
        # 将每个词语切分（包括标点符号）放在列表中。一个词一个列表，列表元素为每个字
        # 当列表长度为1的时候，如 '的'字，那么我们就认为状态为S
        # 当列表长度为2的时候，如'迈向'，我们认为'迈'为B，'向'为E
        # 当长度为3以上的时候，如'实事求是'，我们认为'实'为B，'事求'两个字均为M，'是'为E
        with open(filename, encoding='utf-8') as f:
            for line in f.readlines():
                wordStatus = []                     # 保存该行所有单词的状态
                words = line.strip().split()        # 去除前后空格并将词语分开
                for i, word in enumerate(words):
                    # 更新观测概率矩阵B
                    if len(word) == 1:              # 长度为1的词
                        status = 'S'
                        num = status2num[status]    # 状态在A, B中对应的位置
                        code = ord(word)            # 该字对应的编码
                        B[num, code] += 1           # 先统计频数
                    else:                           # 长度大于等于2的词
                        status = 'B' + 'M' * (len(word) - 2) + 'E'
                        for s in range(len(word)):
                            num = status2num[status[s]]
                            code = ord(word[s])
                            B[num, code] += 1
                    # 更新初始状态概率向量pi
                    if i == 0:
                        num = status2num[status[0]]
                        pi[num] += 1
                    # 将一行（一句话）的每一个状态保存在列表（状态序列）中
                    wordStatus.extend(status)
                # 更新状态转移矩阵A
                for i in range(1, len(wordStatus)):
                    num_t1 = status2num[wordStatus[i - 1]]
                    num_t2 = status2num[wordStatus[i]]
                    A[num_t1, num_t2] += 1

        # 频率转化概率
        # 如果句子较长，许多个较小的数值连乘，容易造成下溢。对于这种情况，我们常常使用log函数解决。
        # 对于没有出现的词语，导致矩阵对应位置0，所以我们需要给每一个0的位置加上一个极小值（-3.14e+100)。
        # 计算状态转移矩阵A
        for i in range(len(A)):
            total = sum(A[i])
            for j in range(len(A)):
                if A[i, j] == 0:
                    A[i, j] = -3.14e+100
                else:
                    A[i, j] = np.log(A[i, j] / total)
        # 计算观测概率矩阵B
        for i in range(len(B)):
            total = sum(B[i])
            for j in range(len(B[i])):
                if B[i, j] == 0:
                    B[i, j] = -3.14e+100
                else:
                    B[i, j] = np.log(B[i, j] / total)
        # 计算初始状态概率向量pi
        pi[pi != 0] = np.log(pi[pi != 0] / sum(pi))
        pi[pi == 0] = -3.14e+100

        self.A = A
        self.B = B
        self.pi = pi

    @staticmethod
    def load_article(filename):
        """
        将测试集文章转化为列表的形式。每句话是一行
        :param filename: 测试文章文件名
        :return: 转化为列表的文章
        """
        with open(filename, encoding='utf-8') as f:
            test_article = []
            for line in f.readlines():
                line = line.strip()       # 去除空格，以及换行符
                test_article.append(line)
        return test_article

    def word_partition(self, article):
        """
        使用维比特算法得到状态序列，即可进行分词。
        :param article: 需要分词的文章，以列表的形式传入，每个元素是一行
        :return: 分词后的文章article_partition
        """
        article_partition = []        # 保存分词后的文章

        for line in article:
            T = len(line)
            delta = np.zeros((T, 4))     # delta的长度为文章的每一行（每一句）的长度，每个位置有4种状态
            psi = np.zeros((T, 4))
            # 初始化
            delta[0] = self.pi + self.B[:, ord(line[0])]    # 由于之前进行了log处理，所以这里是“+”
            for t in range(1, len(line)):
                for i in range(4):
                    delta[t, i] = max(delta[t - 1] + self.A[:, i]) + self.B[i][ord(line[t])]
                    psi[t, i] = (delta[t - 1] + self.A[:, i]).argmax()

            # 最优路径回溯，得到状态序列
            i_T = delta[-1].argmax()
            status = [i_T]              # 保存最优状态序列
            for t in range(len(delta) - 2, -1, -1):
                i_t = psi[t + 1, status[0]]         # status[0]为i_(t-1)
                status.insert(0, int(i_t))

            # 分词
            line_partition = ''
            for t in range(len(line)):
                line_partition += line[t]
                # 状态为'S'或'E'的后面接'|'，句子结尾不用'|'
                if (status[t] == 2 or status[t] == 3) and t != len(line) - 1:
                    line_partition += '|'
            article_partition.append(line_partition)
        return article_partition


if __name__ == '__main__':
    # 用人民日报1998语料库训练模型
    print('用人民日报1998语料库训练模型......')
    print('')
    train_filename = 'HMMTrainSet.txt'
    hmm = HMM()
    hmm.fit(train_filename)

    # 测试模型
    print('================测试文章分词==================')
    article = hmm.load_article('news.txt')
    article_patition = hmm.word_partition(article)
    print(article_patition)
    print('')

    # 自定义测试
    print('=================自定义测试==================')
    line_num = int(input('请输入测试语句行数'))
    lines = []
    for i in range(1, line_num + 1):
        line = input('请输入第{}句语句：'.format(i))
        lines.append(line)
    lines_partition = hmm.word_partition(lines)
    print(lines_partition)

