# lh_ml

### 简介
为了更好地理解《统计学习方法》中的算法，所以用python实现一遍，顺便记录过程中遇到的问题和最后的解决方法，可能有很多细节问题，就先不深究啦。

### 章节

#### 第二章 感知机(Perceptron)
+ 代码文件&ensp;[perceptron.py](https://github.com/ttt256/lh_ml/blob/master/Perceptron/perceptron.py)
	+ 感知机学习算法原始形式、对偶形式的实现
	+ 用 sklearn 自带的数据集 iris dataset 来测试
	+ 选两个特征可视化
	+ 用 sklearn 直接构造感知机

#### 第三章 k近邻法(K Nearest Neighbors)
+ 代码文件&ensp;[KNN.py](https://github.com/ttt256/lh_ml/blob/master/KNearestNeighbors/KNN.py)
	+ 直接遍历所有元素来实现 KNN（K近邻法）
+ 代码文件&ensp;[kdtree_KNN.py](https://github.com/ttt256/lh_ml/blob/master/KNearestNeighbors/kdtree_KNN.py)
	+ 用 kd 树实现 KNN （最近邻搜索和K近邻搜索）
	+ 用 iris dataset 来测试
	+ 选取两个特征进行可视化
	+ 用 sklearn 直接构造 KNN 模型
	
	不用 kd 树的 KNN 很好实现，但每预测一个输入实例，都需要计算该输入实例与每一个训练实例的距离，所以构造 kd 树来缩小搜索的时间复杂度，
	但我刚开始不知道访问到叶节点后怎样回退到父节点，后来发现其实这个就相当于一个二叉树的中序遍历，用递归就可以实现了。
	另外，书上只写了用 kd 树的最近邻搜索算法，我又实现了一个用 kd 树的 K 近邻搜索算法，思路基本一样，唯一不同的地方在于这里不是保存最近邻点，而是保存最近“ k 近邻点集”。
	如果“当前 k 近邻点集”元素数量小于 k ，或者目标点与当前节点距离小于与“当前 k 近邻点集”中最远点的距离，就将该节点插入或替换“当前 k 近邻点集”。
	我先用字典来存储 k 个近邻点，但是在可视化的过程中，我发现找出的 k 个点并不是最近的 k 个点，这是因为当两个点的距离相等的时候，字典只保存了后一个点，所以最后我还是选用了列表来存储 k 个近邻点。

#### 第四章 朴素贝叶斯(Naive Bayes)
书上只写了特征为离散型变量的做法，但是很多特征不是离散型变量，
如果特征为离散型变量，直接按书计算条件概率即可；如果特征为连续型变量，可以考虑把连续型变量化为离散型，或者用其概率密度来计算。
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
+ 代码文件&ensp;[Spam_bag_of_words.py](https://github.com/ttt256/lh_ml/blob/master/NaiveBayes/Spam_bag_of_words.py)
	+ 实现二分类的多项式型朴素贝叶斯模型
	+ 实现 bag of words（两种方法：从头实现 + 调用 CountVectorizer ）
	+ 用 MultinomialNB 直接构造
+ 代码文件&ensp;[naive_Bayes_guassian.py](https://github.com/ttt256/lh_ml/blob/master/NaiveBayes/naive_Bayes_guassian.py)
	+ 实现高斯型朴素贝叶斯（特征均为连续变量）
	+ 用 sklearn 直接构造 GaussianNB
	+ 用 iris dataset 来测试

	高斯型朴素贝叶斯的假定是特征 X 关于 Y 的条件分布是正态的，但是我并没有检验 iris 的特征是否满足条件，所以并不一定正确，
	不过预测的精确度还可以，这里主要是为了说明连续型特征的处理方法，所以就不证明了，如果连续型特征服从其他分布，用其他分布的密度函数应该就可以了。
+ 数据文件&ensp;[mushrooms.txt](https://github.com/ttt256/lh_ml/blob/master/NaiveBayes/mushrooms.txt)
+ 数据文件&ensp;[SMSSpamCollection.txt](https://github.com/ttt256/lh_ml/blob/master/NaiveBayes/SMSSpamCollection.txt)

#### 第五章 决策树(Decision Tree)
+ 代码文件&ensp;[decision_tree_ID3.py](https://github.com/ttt256/lh_ml/blob/master/DecisionTree/decision_tree_ID3.py)
	+ 实现决策树 ID3 算法
	+ 用书上的例 5.1 数据测试
	+ 用 mushrooms dataset 来测试
	+ 直接调用 sklearn 中的 DecisionTreeClassifier
	
	C4.5 算法和 ID3 算法基本一样，加个计算特征A的熵的函数就行，所以就不实现了。
	
#### 第六章 逻辑斯蒂回归(Logistic Regression)和最大熵模型(Maximum Entropy Model)
+ 代码文件&ensp;[logistic_regression.py](https://github.com/ttt256/lh_ml/blob/master/LogisticRegression/logistic_regression.py)
	+ 实现 LR 模型（ mini-batch 梯度下降、正则化），二分类
	+ 使用 Census-Income (KDD) Dataset ，预测个人收入是否超过 $50000。训练集有 54256 个数据，特征有 510 个
	+ 特征标准化
	+ 用 k-fold cross validation 调参
	+ 直接调用 sklearn 中的 LogisticRegression
	
	第一次我没有加正则化，算出来的准确率为 0.88181956 ，并不高，对 X_test 的预测上传到 kaggle 后显示的准确率也只有 0.88842，模型对训练集预测的准确率都不高，说明不是过拟合的问题。
	但我还是试着加了正则项，并用 k-fold 交叉验证选取 lambda 的值，但是选出来的 lambda 几乎等于 0，我觉得可能有两个原因，一是这里并没有过拟合，这一点从第一次不加正则项的结果也可以看出来；
	二是我给损失函数加的 L2 正则项没有除以样本量，虽然从理论上是没问题的，但是这个也可能导致 lambda 取值要小一些。
	为了找到准确率不高的原因，我试着用两种方法进行了特征选择：1. 直接剔除方差过小的特征，这种方法很简单直接；2. 用信息增益，正好用在上一章中刚接触过的 ID3 。
	但是这两种方法都基本上对准确率的提升没太大作用，可能有一点点提升，但在测试集中还是差不多，可能是这个数据集适合其他模型，也可能有其他的优化方法，我学到后面了再来看吧。
+ 代码文件&ensp;[probabilistic_generative_model.py](https://github.com/ttt256/lh_ml/blob/master/LogisticRegression/probabilistic_generative_model.py)
	+ 实现 probabilistic generative model
	+ 对 Census-Income (KDD) Dataset 进行预测
	
	分类还可以用概率生成模型（如果有特征条件独立就是朴素贝叶斯），所以我实现了一个概率生成模型。概率生成模型有可解析的最优解，因此不需要梯度下降等方法，直接计算公式就可以了，
	但用这个模型预测 Census-Income Dataset 的准确率只有 0.6781833 。
	LR 和 probabilistic generative model 看上去很像，但是 LR 是判别模型（Discriminative model），不关心数据集的分布，只需要学习得到分类的规则，
	而生成模型对数据集的分布有严格的假设，比如我实现的这个模型就假设了数据服从高斯分布。
	另一方面，这个模型中需要计算协方差矩阵的逆，在计算逆的过程中，因为这个协方差矩阵非常接近奇异阵，就用 SVD 分解计算了伪逆，
	由于SVD 分解不是唯一的，所以我和别人因为 SVD 分解的结果不一样，导致最后的结果也不一样，特征高度相关对 LR 的影响并不大，但对概率生成模型可能有影响。
+ 数据文件&ensp;[X_train](https://github.com/ttt256/lh_ml/blob/master/LogisticRegression/X_train),
[Y_train](https://github.com/ttt256/lh_ml/blob/master/LogisticRegression/Y_train),
[X_test](https://github.com/ttt256/lh_ml/blob/master/LogisticRegression/X_test)