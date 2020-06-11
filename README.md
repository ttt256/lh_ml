# lh_ml

## 简介 
针对李航《统计学习方法》，主要进行了以下任务：
1. 自己用 python 实现一遍算法，更好地理解书中算法
2. 调用 sklearn 中相关包，熟悉 sklearn
3. 拟合测试集，比较自己实现的模型和 sklearn 中的模型拟合效果
4. 部分可视化，熟悉 matplotlib.pyplot
5. 记录实现过程中遇到的问题

## 章节

### 第二章 感知机(Perceptron)
+ #### 代码文件&ensp;[perceptron.py](https://github.com/ttt256/lh_ml/blob/master/Perceptron/perceptron.py)
	+ 感知机学习算法原始形式、对偶形式的实现
	+ 用 sklearn 自带的数据集 iris dataset 来测试
	+ 选两个特征可视化
	+ 用 sklearn 直接构造感知机

### 第三章 k近邻法(K Nearest Neighbors)
+ #### 代码文件&ensp;[KNN.py](https://github.com/ttt256/lh_ml/blob/master/KNearestNeighbors/KNN.py)
	+ 直接遍历所有元素来实现 KNN（K近邻法）
+ #### 代码文件&ensp;[kdtree_KNN.py](https://github.com/ttt256/lh_ml/blob/master/KNearestNeighbors/kdtree_KNN.py)
	+ 用 kd 树实现 KNN （最近邻搜索和K近邻搜索）
	+ 用 iris dataset 来测试
	+ 选取两个特征进行可视化
	+ 用 sklearn 直接构造 KNN 模型

##### note：
1. kd 树缩小搜索的时间复杂度，要用到回溯算法，用二叉树的中序遍历实现即可。
2. 书上只写了 kd 树的最近邻搜索算法，用一个变量保存最近邻点即可； kd 树的 K 近邻算法需要用一个列表保存“最近 K 近邻点集”，注意：因为数据集有重复数据，用简单的字典会引起一些错误。

### 第四章 朴素贝叶斯(Naive Bayes)
+ #### 代码文件&ensp;[naive_Bayes_discrete.py](https://github.com/ttt256/lh_ml/blob/master/NaiveBayes/naive_Bayes_discrete.py)
	+ 实现多分类的朴素贝叶斯模型（特征均为离散型变量）
	+ 用书上例4.1题的数据测试
	+ 用 iris dataset 来测试，因为 iris 的特征都是连续变量，所以这里考虑将 iris 的特征化为离散型
	+ 用 mushrooms dataset 来测试， mushrooms dataset 有 8124 个数据， 22 个特征，这个数据集的特点是，所有特征和标签都为离散型变量，所以选用这个数据集来测试

##### note：
1. 如果特征为离散型变量，直接按书计算条件概率即可；如果特征为连续型变量，可以考虑把连续型变量化为离散型，或者用其概率密度来计算。
2. 这个代码有两个问题：第一是我是用字典来储存各个条件概率的，假设 y 有 K 类， X 有 j 个特征，平均每个特征有 S 个类型，时间和空间复杂度都会达到 O(KSj) ，虽然这三个测试集没有问题，但是朴素贝叶斯一般用于文本分类，而文本分类的维数都很大；第二是假如训练集中 (x=1, y=0) 的个数为0，那么在 p_x_y 中， P(x=1|y=0) 并没有计算，且在 P(x|y=0) 的计算中， x 的种类 S 少记了一种。对于没有记入字典的 P(x=1|y=0) ，说明在样本中这个事件没有发生，我还是直接取了 0 ，所以虽然套公式的时候用了拉普拉斯平滑，但最后并没有起到拉普拉斯平滑的作用，这是一个有问题的程序。
+ #### 代码文件&ensp;[naive_Bayes_discrete_update.py](https://github.com/ttt256/lh_ml/blob/master/NaiveBayes/naive_Bayes_discrete_update.py)
	+ 用 pandas 的 groupby 实现多分类的朴素贝叶斯（离散特征）
	+ 用上面的三个数据集测试
	+ 用 sklearn 中的 MultinomialNB 直接构造模型

##### note：
1. 使用 pandas 的 groupby 功能，并用 fillna 填充缺失值，解决了上个代码的两个问题。
2. sklearn 中的 MultinomialNB 模型要求特征为数值，但不能简单将特征化为数值型，而要化为 OneHotEncoder 形式（注意为什么要化为 OneHot）。
3. 实现的模型和 sklearn 中的模型在三个数据集上的测试准确率一致。
+ #### 代码文件&ensp;[Spam_set_of_words.py](https://github.com/ttt256/lh_ml/blob/master/NaiveBayes/Spam_set_of_words.py)
	+ 实现 set of words（转小写、去标点、创建词表、创建词向量）
	+ 调用上个文件中实现的模型进行垃圾短信分类

+ #### 代码文件&ensp;[Spam_bag_of_words.py](https://github.com/ttt256/lh_ml/blob/master/NaiveBayes/Spam_bag_of_words.py)
	+ 实现二分类的多项式型朴素贝叶斯模型
	+ 实现 bag of words（两种方法：从头实现 + 调用 CountVectorizer ）
	+ 用 MultinomialNB 直接构造

##### note：
1. 文本分析特征维数大，调用上面实现的模型速度慢，重新实现多项式型朴素贝叶斯。
+ #### 代码文件&ensp;[naive_Bayes_guassian.py](https://github.com/ttt256/lh_ml/blob/master/NaiveBayes/naive_Bayes_guassian.py)
	+ 实现高斯型朴素贝叶斯（特征均为连续变量）
	+ 用 sklearn 直接构造 GaussianNB
	+ 用 iris dataset 来测试

##### note：
1. 高斯型朴素贝叶斯的假定是特征 X 关于 Y 的条件分布是正态的，但是我并没有检验 iris 的特征是否满足条件，所以并不一定正确。
2. 如果连续型特征服从其他分布，用其他分布的密度函数即可。
+ #### 数据文件&ensp;[mushrooms.txt](https://github.com/ttt256/lh_ml/blob/master/NaiveBayes/mushrooms.txt)
+ #### 数据文件&ensp;[SMSSpamCollection.txt](https://github.com/ttt256/lh_ml/blob/master/NaiveBayes/SMSSpamCollection.txt)

### 第五章 决策树(Decision Tree)
+ #### 代码文件&ensp;[decision_tree_ID3.py](https://github.com/ttt256/lh_ml/blob/master/DecisionTree_and_GBDT/decision_tree_ID3.py)
	+ 实现决策树 ID3 算法
	+ 用书上的例 5.1 数据测试
	+ 用 mushrooms dataset 来测试
	+ 直接调用 sklearn 中的 DecisionTreeClassifier
	
+ #### 代码文件&ensp;[decision_tree_C4.5.py](https://github.com/ttt256/lh_ml/blob/master/DecisionTree_and_GBDT/decision_tree_C4.5.py)
	+ 实现决策树 C4.5算法
	+ 用书上的例 5.1 数据测试
	+ 用 mushrooms dataset 来测试

+ #### 代码文件&ensp;[gbdt.py](https://github.com/ttt256/lh_ml/blob/master/DecisionTree_and_GBDT/gbdt.py)
	+ 实现 GBDT ，其中的树为 CART 回归树
	+ 对波士顿房价数据进行预测，选择均方误差 MSE 和 拟合系数 R2score 作为模型评价指标
	
+ #### 数据文件&ensp;[boston_house_prices.csv](https://github.com/ttt256/lh_ml/blob/master/DecisionTree_and_GBDT/boston_house_prices.csv)


### 第六章 逻辑斯蒂回归(Logistic Regression)和最大熵模型(Maximum Entropy Model)
+ #### 代码文件&ensp;[logistic_regression.py](https://github.com/ttt256/lh_ml/blob/master/LogisticRegression/logistic_regression.py)
	+ 实现 LR 模型（ mini-batch 梯度下降、正则化），二分类
	+ 使用 Census-Income (KDD) Dataset ，预测个人收入是否超过 $50000。训练集有 54256 个数据，特征有 510 个
	+ 特征标准化
	+ 用 k-fold cross validation 调参
	+ 直接调用 sklearn 中的 LogisticRegression

##### note：
1. 特征选择：（1）直接剔除方差过小的特征，这种方法很简单直接；（2） 用信息增益，正好用在上一章中刚接触过的 ID3 。在该训练集上效果不好。

+ #### 代码文件&ensp;[probabilistic_generative_model.py](https://github.com/ttt256/lh_ml/blob/master/LogisticRegression/probabilistic_generative_model.py)
	+ 实现 probabilistic generative model
	+ 对 Census-Income (KDD) Dataset 进行预测

##### note：
1. 概率生成模型有可解析的最优解，因此不需要梯度下降等方法，直接计算公式即可。
2. 比较概率生成模型和 LR （判别模型）在该数据集上的效果，分析结果不同的原因。
3. 概率生成模型中需要计算协方差矩阵的逆，在计算逆的过程中，因为这个协方差矩阵非常接近奇异阵，就用 SVD 分解计算了伪逆。
4. 由于SVD 分解不是唯一的，所以我和别人因为 SVD 分解的结果不一样，导致最后的结果也不一样，特征高度相关对 LR 的影响并不大，但对概率生成模型可能有影响。
+ #### 数据文件&ensp;[X_train](https://github.com/ttt256/lh_ml/blob/master/LogisticRegression/X_train),[Y_train](https://github.com/ttt256/lh_ml/blob/master/LogisticRegression/Y_train),[X_test](https://github.com/ttt256/lh_ml/blob/master/LogisticRegression/X_test)


### 第十章 隐马尔可夫模型(Hidden Markov Model)
+ #### 代码文件&ensp;[hmm.py](https://github.com/ttt256/lh_ml/blob/master/HMM/hmm.py)
	+ 实现HMM模型
	+ 用人民日报1998语料库训练模型
	+ 对新闻和输入语句进行测试
	
##### note：
1. HMM典型模型是一个五元组：
	状态值集合：(B, M, E, S): {B:begin, M:middle, E:end, S:single}
	观察值集合：在ord中，中文编码大小为65536，总共4个状态，所以B矩阵4x65536
	转移概率矩阵A
	观察概率矩阵B
	初值状态概率向量pi
2. 用HMM进行分词：
	比如：小明硕士毕业于中国科学院计算所
	输出的状态序列为：BEBEBMEBEBMEBES
	根据这个状态序列我们可以进行切词：BE/BE/BME/BE/BME/BE/S
	所以切词结果如下：小明/硕士/毕业于/中国/科学院/计算/所
3. 如果句子较长，许多个较小的数值连乘，容易造成下溢。对于这种情况，我们常常使用log函数解决。对于没有出现的词语，矩阵对应的位置为0，因为log0不存在，所以我们需要给每一个0的位置加上一个极小值。
+ #### 人民日报1998语料库&ensp;[HMMTrainSet.txt](https://github.com/ttt256/lh_ml/blob/master/HMM/HMMTrainSet.txt)
+ #### 测试文章&ensp;[news.txt](https://github.com/ttt256/lh_ml/blob/master/HMM/news.txt)
