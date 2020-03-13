# lh_ml

### 简介
为了更好地理解《统计学习方法》中的算法，所以用python实现一遍，顺便记录过程中遇到的问题和最后的解决方法，可能有很多细节问题，就不先深究啦。

### 章节
#### 第二章 感知机(Perceptron)
+ 代码文件&ensp;[perceptron.py](https://github.com/ttt256/lh_ml/blob/master/Perceptron/perceptron.py)
	+ 感知机学习算法原始形式、对偶形式的实现
	+ 用 sklearn 直接构造感知机
	+ 用 sklearn 自带的数据集 iris dataset 来测试

#### 第三章 k近邻法(KNearestNeighbors)
+ 代码文件&ensp;[KNN.py](https://github.com/ttt256/lh_ml/blob/master/KNearestNeighbors/KNN.py)
	+ 直接遍历所有元素来实现 KNN
+ 代码文件&ensp;[kdtree_KNN.py](https://github.com/ttt256/lh_ml/blob/master/KNearestNeighbors/kdtree_KNN.py)
	+ 用 kd 树实现 KNN 
	+ 用 sklearn 直接构造 KNN 模型
	+ 用 iris dataset 来测试

#### 第四章 朴素贝叶斯(NaiveBayes)
书上只写了特征为离散型变量的做法，但是很多特征不是离散型变量，
如果特征为离散型变量，直接按书计算条件概率即可；如果特征为连续型变量，可以考虑把连续性变量化为离散型，或者用其概率密度来计算。
+ 代码文件&ensp;[naive_Bayes_discrete.py](https://github.com/ttt256/lh_ml/blob/master/NaiveBayes/naive_Bayes_discrete.py)
	+ 实现多分类的朴素贝叶斯模型（特征均为离散型变量）
	+ 用书上例4.1题的数据测试
	+ 用 iris dataset 来测试，因为 iris 的特征都是连续变量，所以这里考虑将 iris 的特征化为离散型
	+ 用 mushrooms dataset 来测试， mushrooms dataset 有 8124 个数据， 22 个特征，这个数据集的特点是，所有特征和标签都为离散型变量，所以选用这个数据集来测试

这个代码有两个问题：第一是我是用字典来储存各个条件概率的，假设 y 有 K 类， X 有 j 个特征，平均每个特征有 S 个类型，时间和空间复杂度都会达到 O(KSj) ，
虽然这三个测试集没有问题，但是朴素贝叶斯一般用于文本分类，而文本分类的维数都很大；
第二是假如训练集中 (x=1, y=0) 的个数为0，那么在 p_x_y 中， P(x=1|y=0) 并没有计算，
且在 P(x|y=0) 的计算中， x 的种类 S 少记了一种。对于没有记入字典的 P(x=1|y=0) ，说明在样本中这个事件没有发生，我还是直接取了 0 ，
所以虽然套公式的时候用了拉普拉斯平滑，但最后并没有起到拉普拉斯平滑的作用，这是一个有问题的程序。
虽然三个测试的正确率很高，但是为了提高效率以及正确使用拉普拉斯平滑，我又尝试用 pandas 中的 groupby 实现了一次。
	
+ 代码文件&ensp;[naive_Bayes_discrete_update.py](https://github.com/ttt256/lh_ml/blob/master/NaiveBayes/naive_Bayes_discrete_update.py)
	+ 用 pandas 的 groupby 实现多分类的朴素贝叶斯（离散特征）
	+ 用上面的三个数据集测试
	+ 对特征和标签数值化（ MultinomialNB 要求输入的 X 为数值）
	+ 用 sklearn 中的 MultinomialNB 直接构造模型（精确度低）
	+ 使用 OneHotEncoder 将数据化为独热编码的形式
	+ 再调用 MultinomialNB 直接构造模型（精确度和实现的模型一致）

这个代码中我用了 pandas 的 groupby 功能，并用 fillna 填充了缺失值，解决了上个代码中存在条件概率为 0 的情况。
因为 sklearn 中的 MultinomialNB 模型要求特征为数值，所以我先将特征数值化，再套入 MultinomialNB ，
但是这个时候不仅预测 iris 和 mushrooms 数据集的精确度很低，连书上例 4.1 的简单例子预测结果也不一样了。
最后发现是因为 MultinomialNB 适合每个特征服从多项分布的数据集，对于不服从多项分布的离散特征，要先化成独热编码的形式，
化为独热编码的形式之后，再调用 MultinomialNB ，得到三个数据集的测试精确度都和实现的模型精确度一样。
+ 代码文件&ensp;[Spam_set_of_words.py](https://github.com/ttt256/lh_ml/blob/master/NaiveBayes/Spam_set_of_words.py)
	+ 实现 set of words（转小写、去标点、创建词表、创建词向量）
	+ 调用上个文件中实现的模型进行垃圾短信分类
	
词集模型（单词出现为 1 ，没出现为 0 ），词袋模型（单词频数）。
词袋模型的每个特征关于标签 Y 的条件分布都是多项分布，适合多项式型的朴素贝叶斯，而上个文件中实现的模型只能进行词集模型，
另外由于这个模型依然很慢很慢，不是一个好的模型，所以我参考别人的代码又重新实现了一个。
+ 代码文件&ensp[Spam_bag_of_words.py](https://github.com/ttt256/lh_ml/blob/master/NaiveBayes/Spam_bag_of_words.py)
	+ 实现二分类的多项式型朴素贝叶斯模型
	+ 实现 bag of words（两种方法：从头实现 + 调用 CountVectorizer ）
	+ 用 MultinomialNB 直接构造
+ 代码文件&ensp;[naive_Bayes_guassian.py](https://github.com/ttt256/lh_ml/blob/master/NaiveBayes/naive_Bayes_guassian.py)
	+ 高斯型朴素贝叶斯（特征均为连续变量）
	+ 用 sklearn 直接构造 GaussianNB
	+ 用 iris dataset 来测试

高斯型朴素贝叶斯的假定是特征 X 关于 Y 的条件分布是正态的，但是我并没有检验 iris 的特征是否满足条件，所以并不一定正确，
不过预测的精确度还可以，这里主要是为了说明连续型特征的处理方法，所以就不证明了，如果连续型特征服从其他分布，用其他分布的密度函数应该就可以了。
+ 数据文件&ensp;[mushrooms.txt](https://github.com/ttt256/lh_ml/blob/master/NaiveBayes/mushrooms.txt)
+ 数据文件&ensp;[SMSSpamCollection.txt](https://github.com/ttt256/lh_ml/blob/master/NaiveBayes/SMSSpamCollection.txt)
