#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# __author__ = 'Frank'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 导入数据
df = pd.read_csv('D:\\data analysis\\PycharmProjects\\datalearning\\SVM\\breast_cancer_data-master\\data.csv')
# 数据探索
pd.set_option('display.max_columns', None)
print(df.info())
print(df.head())
print(df.describe())
print(df.columns)
# 数据清洗
# 删除id列
df.drop('id', axis = 1, inplace = True)
# 将B良性替换为0，M恶性替换为1
df.diagnosis = df.diagnosis.map({'B': 0, 'M': 1})
# 将诊断结果可视化
sns.countplot(df.diagnosis)
plt.show()
# 利用所有特征创建模型
# 抽取30%的数据做测试集，其余做训练集
train_x, test_x, train_y, test_y = train_test_split(df.iloc[:, 1:], df.diagnosis, test_size = 0.3)
# 采用Z_score规范化数据
scale = StandardScaler()
train_x = scale.fit_transform(train_x)
test_x = scale.transform(test_x)
# 训练模型
model = svm.LinearSVC()
model.fit(train_x, train_y)
# 测试模型
predict_y = model.predict(test_x)
acc_score = accuracy_score(test_y, predict_y)
print(acc_score)
