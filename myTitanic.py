# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from collections import Counter

# import sys
# # reload(sys)
# # sys.setdefaultencoding('utf-8')

# pandas dataframe数据全部输出，数据太多也不用省略号表示。
pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)  #当consel中输出的列超过1000的时候才会换行

# 读取训练数据集
train = pd.read_csv('train.csv')
# 读取测试数据集
test = pd.read_csv('test.csv')

print('训练数据集:')
print(train.shape)
print('测试数据集:')
print(test.shape)

# 合并数据集，方便同时对两个数据集进行清洗
# add 'sort=True'
full = train.append(test,ignore_index=True,sort=True)

print('合并后的数据集:')
print(full.shape)

# ## 2、查看数据集信息

# 查看数据
print full.head()
print full.tail()

# 获取数据类型列的描述统计信息
print full.describe()

# 查看每一列的数据类型，和数据总数
print full.info()

# ## 3、数据清洗

# ### （1）处理缺失值

# 数值型缺失值处理

'''
年龄（Age）和船票价格（Fare）存在缺失值且都为浮点型数据，
可以使用平均值填充的方法填充缺失值
'''
# 年龄（Age）
full['Age'] = full['Age'].fillna(full['Age'].mean())
# 船票价格（Fare）
full['Fare'] = full['Fare'].fillna(full['Fare'].mean())

# 字符串类型缺失值处理

# 填充客舱号（Cabin）,先查看里面的数据
print full['Cabin'].head()

'''
客舱号（Cabin）这一列缺失数值较多且无规律可循，
直接填充为U，表示未知（Unknown）
'''
full['Cabin'] = full['Cabin'].fillna('U')

# 有类别数据缺失值处理

'''
先查看 前5行的信息，看看数据的样子
出发地点：S=英国 南安普顿Southampton
途径地点1：C=法国 瑟堡市Cherbourg
途径地点2：Q=爱尔兰 昆士敦Queenstown
'''
print full['Embarked'].head()

print Counter(full['Embarked'])

'''
由前面的信息可知，登船港口（Embarked）这一列只有两个缺失值，
将缺失值填充为最频繁出现的值：S=英国 南安普顿Southampton
'''
full['Embarked'] = full['Embarked'].fillna('S')


# 查看最终缺失值处理情况
print full.info()

# ### （2）特征提取

# #### a.分类数据特征提取：性别

'''
将性别的值映射为数值
男（male）对应数值 1
女（female）对应数值 0
'''
sex_mapDict = {'male':1,
               'female':0}
# map函数：对于Series每个数据应用自定义函数计算
full['Sex'] = full['Sex'].map(sex_mapDict)
print full.head()

# #### b.分类数据特征提取：登船港口

# 查看该类数据内容
print full['Embarked'].head()

# In[15]:

# 存放提取后的特征
embarkedDf = pd.DataFrame()

# 使用get_dummies进行one-hot编码，列名前缀为Embarked
embarkedDf = pd.get_dummies(full['Embarked'],prefix='Embarked')
embarkedDf.head()

# 添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full,embarkedDf],axis=1)
'''
因为已经使用登船港口（embarkedDf）进行了one-hot编码产生虚拟变量（dummy variables）
所以这里把登船港口（embarkedDf）删除
'''
full.drop('Embarked',axis=1,inplace=True)
print full.head()

# #### c.分类数据特征提取：客舱等级

# 存放提取后的特征
pclassDf = pd.DataFrame()

# 使用get_dummies进行one-hot编码，列名前缀为Pclass
pclassDf = pd.get_dummies(full['Pclass'],prefix='Pclass')
print pclassDf.head()

# 添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full,pclassDf],axis=1)

# 删除客舱等级（Pclass）这一列
full.drop('Pclass',axis=1,inplace=True)
print full.head()

# #### d.分类数据特征提取：姓名

print full['Name'].head()

# 定义函数： 从姓名中获取头衔
# split()通过指定分隔符对字符串进行切片
def getTitle(name):           # Braund, Mr. Owen Harris
    str1 = name.split(',')[1] # Mr. Owen Harris
    str2 = str1.split('.')[0] # Mr
    # strip()方法用于移除字符串头尾指定的字符（默认为空格）
    str3 = str2.strip()
    return str3

# 存放提取后的特征
titleDf = pd.DataFrame()
# map函数：对于Series每个数据应用自定义函数计算
titleDf['Title'] = full['Name'].map(getTitle)
print titleDf.head()

print Counter(titleDf['Title'])

# In[21]:

# 姓名中头衔字符串与定义头衔类别的映射关系
title_mapDict = {
    'Capt':        'Officer',
    'Col':         'Officer',
    'Major':       'Officer',
    'Jonkheer':    'Royalty',
    'Don':         'Royalty',
    'Sir':         'Royalty',
    'Dr':          'Officer',
    'Rev':         'Officer',
    'the Countess':'Royalty',
    'Dona':        'Royalty',
    'Mme':         'Mrs',
    'Mlle':        'Miss',
    'Ms':          'Mrs',
    'Mr':          'Mr',
    'Mrs':         'Mrs',
    'Miss':        'Miss',
    'Master':      'Master',
    'Lady':        'Royalty'
}

# map函数：对于Series每个数据应用自定义函数计算
titleDf['Title'] = titleDf['Title'].map(title_mapDict)
# 使用get_dummies进行one-hot编码
titleDf = pd.get_dummies(titleDf['Title'])
print titleDf.head()

# 添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full,titleDf],axis=1)

# 删除姓名（Name）这一列
full.drop('Name',axis=1,inplace=True)
print full.head()


# #### e.分类数据特征提取：客舱号

print full['Cabin'].head()

# 存放客舱号信息
cabinDf = pd.DataFrame()

'''
客舱号的类别值是首字母，例如：
C85 类别映射为首字母C
'''
full['Cabin'] = full['Cabin'].map(lambda c:c[0]) # 定义匿名函数lambda，用于查找首字母

# 使用get_dummies进行one-hot编码，列名前缀为Cabin
cabinDf = pd.get_dummies(full['Cabin'],prefix='Cabin')
print cabinDf.head()

# 添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full,cabinDf],axis=1)

# 删除客舱号（Cabin）这一列
full.drop('Cabin',axis=1,inplace=True)
print full.head()

# #### f.分类数据特征提取：家庭类别

# 存放家庭信息
familyDf = pd.DataFrame()

'''
家庭人数 = 同代直系亲属数（SibSp）+ 不同代直系亲属数（Parch）+ 乘客自己
（乘客自己也属于家庭成员一个，所以要加1）
'''
familyDf['Familysize'] = full['SibSp'] + full['Parch'] + 1

'''
家庭类别：
小家庭Family_Single:家庭人数=1
中等家庭Family_Small:2<=家庭人数<=4
大家庭Family_Large:家庭人数>=5
'''

# if条件为真时返回if前面内容，否则返回0
familyDf['Family_Single'] = familyDf['Familysize'].map(lambda s: 1 if s==1 else 0)
familyDf['Family_Small']  = familyDf['Familysize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
familyDf['Family_Large']  = familyDf['Familysize'].map(lambda s: 1 if 5 <= s else 0)

print familyDf.head()

# In[27]:

# 添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full,familyDf],axis=1)
print full.head()

# #### g.分类数据特征提取：年龄

# 存放年龄信息
ageDf = pd.DataFrame()

'''
年龄类别：
儿童Child:0<年龄<=6
青少年Teenager：6<年龄<18
青年Youth：18<=年龄<=40
中年Middle_aged：40<年龄<=60
老年Older:60<年龄
'''
# if条件为真时返回if前面内容，否则返回0
ageDf['Child']       = full['Age'].map(lambda a: 1 if 0 < a <= 6 else 0)
ageDf['Teenager']    = full['Age'].map(lambda a: 1 if 6 < a < 18 else 0)
ageDf['Youth']       = full['Age'].map(lambda a: 1 if 18 <= a <= 40 else 0)
ageDf['Middle_aged'] = full['Age'].map(lambda a: 1 if 40 < a <= 60 else 0)
ageDf['Older']       = full['Age'].map(lambda a: 1 if 60 < a else 0)

print ageDf.head()

# 添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full,ageDf],axis=1)

# 删除年龄（Age）这一列
full.drop('Age',axis=1,inplace=True)
print full.head()

# 查看现在已有的特征
print full.shape

# ### （3）特征选择

# 相关性矩阵

# 相关性矩阵
corrDf = full.corr()
print corrDf

'''
查看各个特征与生成情况（Survived）的相关系数，
ascending = False 表示按降序排列
'''

print corrDf['Survived'].sort_values(ascending = False)

# 根据各个特征与生存情况（Survived）的相关系数大小，选择以下几个特征作为模型的输入：

# 头衔（titleDf)、客舱等级（pclassDf）、家庭大小（familyDf）、船票价格（Fare）、性别（Sex）、客舱号（cabinDf）、登船港口（embarkdeDf）

# 特征选择
full_X = pd.concat([titleDf,     # 头衔
                    pclassDf,    # 客舱等级
                    familyDf,    # 家庭大小
                    full['Fare'],# 船票价格
                    full['Sex'], # 性别
                    cabinDf,     # 客舱号
                    embarkedDf   # 登场港口
                   ],axis=1)

print full_X.head()

# ## 4、构建模型

# 使用训练数据和机器学习算法得到一个机器学习模型，再使用测试数据评估模型

# ### （1）建立训练数据集和测试数据集

# 原始数据共有891行
sourceRow = 891

'''
原始数据集sourceRow是从Kaggle下载的训练数据集，可知共有891条数据
从特征集合full_X中提取原始数据集前891行数据时需要减去1，因为行号是从0开始
'''
# 原始数据集：特征
source_X = full_X.loc[0:sourceRow-1,:]
# 原始数据集：标签
source_y = full.loc[0:sourceRow-1,'Survived']

# 测预测数据集：特征
pred_X = full_X.loc[sourceRow:,:]

# 查看原始数据集有多少行
print('原始数据集有多少行：',source_X.shape[0])
# 查看预测数据集有多少行
print('预测数据集有多少行：',pred_X.shape[0])

'''
从原始数据集（source）中拆分出用于模型训练的训练数据集（train）,用于评估模型的测试数据集（test）
train_test_split:是交叉验证中常用的函数，功能是从样本中随机按比例选取train data和test data
train_data:所要划分的样本特征集
train_target:所要划分的样本结果
train_size:样本占比，如果为整数则是样本的数量
'''
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split

# 建立模型所需的训练数据集和测试数据集
train_X,test_X,train_y,test_y = train_test_split(source_X,source_y,train_size=0.8)

# 输出数据集大小
print('原始数据集特征',source_X.shape,
      '训练数据集特征',train_X.shape,
      '测试数据集特征',test_X.shape)

print('原始数据集标签',source_y.shape,
      '训练数据集标签',train_y.shape,
      '测试数据集标签',test_y.shape)

# 查看原始数据集标签
print source_y.head()

# ### （2）选择机器学习算法

# 选择一个机器学习算法，用于模型训练，这里选择逻辑回归（logisic regression）

# 第1步：导入算法
from sklearn.linear_model import LogisticRegression
# 第2步：创建模型：逻辑回归（logisic regression）
model = LogisticRegression()

# 随机森林Random Forests Model
#from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier(n_estimators=100)
#0.7765


# 支持向量机Support Vector Machines
#from sklearn.svm import SVC,LinearSVC
#model = SVC()
#0.78

# Gradient Boosting Classifier
#from sklearn.ensemble import GradientBoostingClassifier
#model = GradientBoostingClassifier()
#0.81

#K-nearest neighbors
#from sklearn.neighbors import KNeighborsClassifier
#model = KNeighborsClassifier(n_neighbors = 3)
#0.75

# 朴素贝叶斯Gaussian Naive Bayes
#from sklearn.naive_bayes import GaussianNB
#model = GaussianNB()
#0.81


# ### （3）训练模型

# 第3步：训练模型
model.fit(train_X,train_y)


# ### （4）评估模型

# 分类问题，score得到的是模型正确率
model.score(test_X,test_y)

# ## 5、实施方案

# 使用预测数据集进行预测结果，并保存到csv文件中，最后上传到Kaggle中

# 使用机器学习模型，对预测数据集中的生存情况进行预测
pred_y = model.predict(pred_X)

'''
生成的预测值是浮点数（0.0,1.0）
但是Kaggle要求提交的结果是整型（0,1）
使用astype对数据类型进行转换
'''
pred_y = pred_y.astype(int)
# 乘客id
passenger_id = full.loc[sourceRow:,'PassengerId']
# 数据框：乘客id，预测生存情况
predDf = pd.DataFrame({'PassengerId':passenger_id, 'Survived':pred_y})
print predDf.shape
print predDf.head()


# 保存结果
predDf.to_csv('titanic_pred.csv',index=False)





