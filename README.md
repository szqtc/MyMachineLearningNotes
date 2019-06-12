机器学习笔记本
============
介绍一些常见的机器学习算法、简单粗糙的Python 3实现，以及它们在已有的工具包中的使用方法。  
*我非计算机专业，初学机器学习，水平尚浅，内容仅供参考*。

目录（暂定）
----------------
1. [线性模型](1_LinearModel.ipynb)
    - 线性模型
        + 线性回归
        + 线性基函数
        + 正则化(岭回归, LASSO)
        + 核技巧
            * 核岭回归(KRR)
        + 局部加权线性回归(LWR)
    - 广义线性模型(GLMs)
        + Logistic回归(LR)
        + Softmax回归
    - 判别式分析
        + 线性判别式分析(LDA)
        + 二次判别式分析(QDA)
        + 核线性判别式分析(KLDA) [TODO]
2. [支持向量机](2_SVM.ipynb)
    - 线性支持向量分类器(SVC)
        + 求解SVC(SMO算法)
    - 核SVC
    - 支持向量回归(SVR)
3. [决策树与集成方法](3_Trees.ipynb)
    - 决策树(DT)
        + 决策树的学习(ID3, C4.5)
            * 树的生成
            * 特征选择
            * 树的修剪
            * 连续值和缺失值的处理
        + CART算法：分类决策树和回归树
    - 集成方法
        + 平均方法
            * Bagging
            * 随机森林(RF)
        + 提升方法
            * AdaBoost
            * 回归问题中的提升方法
            * 梯度提升树(GBDT) [TODO]
            * XGBoost模型 [TODO]
4. 贝叶斯模型与概率图模型 [TODO]
    - 朴素贝叶斯分类器
    - 贝叶斯网
    - 隐Markov模型
    - 条件随机场(CRF)
5. 神经网络与深度学习 [TODO]
    - `tensorflow`基础
    - 感知机
    - 前馈神经网络
    - 卷积神经网络(CNN)
    - 递归神经网络(RNN)
6. 非监督学习 [TODO]
    - 聚类
    - 降维
    

主要参考资料
------------------
1. 机器学习, 周志华, 2016.
2. 机器学习实战, P. Harrington, 2013.
3. 统计学习方法, 李航, 2012.
4. An Introduction to Statistical Learning (ISL), G. James, D. Witten, T. Hastie and R. Tibshirani, 2013.
5. [scikit-learn.org](http://scikit-learn.org)
6. [deeplearning.ai](https://www.deeplearning.ai)

----------------------
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by/4.0/">知识共享署名 4.0 国际许可协议</a>进行许可。