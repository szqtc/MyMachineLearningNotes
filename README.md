机器学习笔记本
============
介绍一些常见的机器学习算法、简单的Python 3实现，以及它们在已有的工具包中的使用方法。  
我非计算机专业，初学机器学习，水平尚浅，内容仅供参考。

目录（暂定）
----------------
1. [线性模型](1_LinearModel.ipynb)
    - 线性模型
        + 线性回归
        + 线性基函数
        + 正则化(岭回归, LASSO)
        + 局部加权线性回归(LWR)
        + 核技巧
            * 核岭回归(KRR) [TODO]
    - 广义线性模型(GLMs)
        + Logistic回归(LR)
        + Softmax回归 [TODO]
    - 判别式分析
        + 线性判别式分析(LDA)
        + 二次判别式分析(QDA)
        + 核线性判别式分析(KLDA) [TODO]
2. [支持向量机](2_SVM.ipynb)
    - 线性支持向量分类器(SVC)
        + 求解SVC(SMO算法)
    - 核SVC
    - 支持向量回归(SVR) [TODO]
3. [决策树](3_Trees.ipynb)
    - 决策树(DT)
        + 决策树的学习(ID3, C4.5)
            * 树的生成
            * 特征选择
            * 树的修剪
            * 其他的技术细节(连续值和缺失值的处理)
        + CART算法：分类决策树和回归树
    - 集成方法
        + 平均方法
            * Bagging
            * 随机森林(RF) [TODO]
        + 提升方法 [TODO]
            * AdaBoost
            * 提升树
4. 神经网络与深度学习 [TODO]
    - `tensorflow`基础与`keras`包
    - 感知机
    - 前馈神经网络
    - 卷积神经网络
    - 递归神经网络
5. 非监督学习 [TODO]
    - 聚类
    - 降维
    

参考资料
------------
1. 西瓜书
2. 机器学习实战
3. 统计学习方法
4. An Introduction to Statistical Learning
5. [scikit-learn.org](http://scikit-learn.org)
6. [deeplearning.ai](https://www.deeplearning.ai)