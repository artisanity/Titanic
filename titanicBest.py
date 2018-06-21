# -*- coding: utf-8 -*-

#系统调用
import os
#数据处理
import pandas as pd
import numpy as np
import random
from numpy import median, mean

#可视化处理
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#模型算法
import sklearn.preprocessing as preprocessing
from sklearn import linear_model, ensemble
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance


#指定默认字体
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, plot_importance

# 解决图标中文乱码问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'
#解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False


# pandas dataframe 数据全部输出，数据列数太多也不用省略号表示。
pd.set_option('display.max_columns',None)
# pandas dataframe 数据全部输出，数据列数多时设置超过一定列数才会换行。
pd.set_option('display.width',1000)  #当consel中输出的列超过1000的时候才会换行

# path = 'D:/AILearning/project/AILearning/Titanic/'
path = 'D:\\AILearning\\project\\AILearning\\Titanic\\'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

# print train.head()
# print train.describe()
# print train.info()
# print train.Name.apply(lambda x: len(x))
# print train.groupby(train.Name.apply(lambda x: len(x)))['Survived'].mean()
# train.groupby(train.Name.apply(lambda x: len(x)))['Survived'].mean().plot()

# plt.plot(train.groupby(train.Name.apply(lambda x: len(x)))['Survived'].mean())
# plt.plot(train.groupby(train['Name'].apply(lambda x: x.split(', ')[1]).apply(lambda x: x.split('.')[0]))['Survived'].mean())
# plt.show()

# print(u'训练数据集：')
# print(train.shape)
# print(u'测试数据集：')
# print(test.shape)

#合并数据集，方便同时对两个数据集进行清洗
# full = train.append(test, ignore_index = True, sort=True)
full = train.append(test, ignore_index = True, sort=False)
# print(u'合并后数据集：')
# print(full.shape)
# print full.info()

full['Name_Len'] = full['Name'].apply(lambda x: len(x))
full['Name_Len'] = pd.qcut(full['Name_Len'],5)
nameLenDf = pd.get_dummies(full.Name_Len,prefix='Name_Len')
full = pd.concat([full,nameLenDf],axis=1).drop('Name_Len',axis=1)

# print full.head()
# print full['Name_Len'].value_counts()
# print full['Fare'].value_counts()

#查看头5行数据
# print full.head()
# print full.tail()
#获取数据类型列的描述统计信息
# print full.describe()
#查看每一列的数据类型和数据总数
# print full.info()

#数据缺失值处理(用平均值填充)
#年龄（Age) ，另一种处理方法：结合title和pclass处理
full['Age']=full['Age'].fillna(full['Age'].mean())

#登船港口（Embarked）：查看里面数据长啥样
'''
出发地点：S=英国南安普顿Southampton
途径地点1：C=法国 瑟堡市Cherbourg
途径地点2：Q=爱尔兰 昆士敦Queenstown
'''
# print full['Embarked'].head()
'''
分类变量Embarked，看一下最常见的类别，用其填充
'''
# print full['Embarked'].value_counts()
'''
从结果来看，S类别最常见。我们将缺失值填充为最频繁出现的值：
S=英国南安普顿Southampton
缺失Embarked信息的乘客的Pclass均为1，且Fare均为80。  
C均值最接近，取C  
'''
# pc1 = list(set(full[(full.Pclass==1) & (full.Embarked=='C')]['Fare'].values))
# pc2 = list(set(full[(full.Pclass==1) & (full.Embarked=='Q')]['Fare'].values))
# pc3 = list(set(full[(full.Pclass==1) & (full.Embarked=='S')]['Fare'].values))
# print median(pc1)
# print median(pc2)
# print median(pc3)
# print mean(pc1)
# print mean(pc2)
# print mean(pc3)

'''
61.3792
90.0
51.4792
83.73401818181819
90.0
66.08961639344264
'''

full['Embarked'] = full['Embarked'].fillna( 'C' )
# full['Embarked'] = full['Embarked'].fillna( 'S' )


'''
从结果来看，S类别最常见。我们将缺失值填充为最频繁出现的值：
S=英国南安普顿Southampton
'''
# full['Embarked'] = full['Embarked'].fillna( 'S' )

#船票价格（Fare),只在test中有一个空值
# full['Fare']=full['Fare'].fillna(full['Fare'].mean())
#船票价格（Fare),补缺值由均值改为众数
full['Fare'] = pd.qcut(full.Fare,3)
# df = pd.get_dummies(full.Fare,prefix='Fare').drop('Fare_(-0.001, 8.662]',axis=1)
fareDf = pd.get_dummies(full.Fare,prefix='Fare')
full = pd.concat([full,fareDf],axis=1).drop('Fare',axis=1)
# print full.head()


#船舱号（Cabin）：查看里面数据长啥样
# print full['Cabin'].head()
#缺失数据比较多，船舱号（Cabin）缺失值填充为U，表示未知（Uknow）
full['Cabin'] = full['Cabin'].fillna( 'U' )

full['Fname'] = full['Name'].apply(lambda x:x.split(',')[0])
full['Familysize'] = full['SibSp']+full['Parch']
dead_female_Fname = list(set(full[(full.Sex=='female') & (full.Age>=12)
                              & (full.Survived==0) & (full.Familysize>1)]['Fname'].values))
survive_male_Fname = list(set(full[(full.Sex=='male') & (full.Age>=12)
                              & (full.Survived==1) & (full.Familysize>1)]['Fname'].values))
full['Dead_female_family'] = np.where(full['Fname'].isin(dead_female_Fname),1,0)
full['Survive_male_family'] = np.where(full['Fname'].isin(survive_male_Fname),1,0)
# full = full.drop(['Name','Fname'],axis=1)
full = full.drop(['Fname'],axis=1)
full = full.drop(['Familysize'],axis=1)

# print full.head()

# print full.info()

sex_mapDict={'male':1, 'female':0}
#map函数：对Series每个数据应用自定义的函数计算
full['Sex']=full['Sex'].map(sex_mapDict)

# print full.head()

#存放提取后的特征
embarkedDf = pd.DataFrame()

'''
使用get_dummies进行one-hot编码，产生虚拟变量（dummy variables)
列名前缀是Embarked
'''
embarkedDf = pd.get_dummies(full['Embarked'], prefix='Embarked')
#添加one-hot编码产生的虚拟变量（dummy variables)到泰坦尼克号数据集full
full = pd.concat([full,embarkedDf], axis=1)

'''
因为已经使用登船港口(Embarked)进行了one-hot编码产生了它的虚拟变量（dummy variables）
所以这里把登船港口(Embarked)删掉
'''
full.drop('Embarked',axis=1,inplace=True)
# print full.head()

#存放提取后的特征
pclassDf = pd.DataFrame()
#使用get_dummies进行one-hot编码，列名前缀是Pclass
pclassDf = pd.get_dummies(full['Pclass'], prefix='Pclass')
#添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full,pclassDf],axis=1)
#删掉客舱等级（Pclass)这一列
full.drop('Pclass',axis=1,inplace=True)
# print full.head()


#从姓名中提出特征：头衔
#先查看姓名数据，看看有什么特征
full[ 'Name' ].head()
#练习从字符串中提取头衔，例如Mr
#split用于字符串分割，返回一个列表
#我们看到姓名中'Braund, Mr. Owen Harris'，逗号前面的是“名”，逗号后面是‘头衔. 姓’
name1='Braund, Mr. Owen Harris'
'''
split用于字符串按分隔符分割，返回一个列表。这里按逗号分隔字符串
也就是字符串'Braund, Mr. Owen Harris'被按分隔符,'拆分成两部分[Braund,Mr. Owen Harris]
你可以把返回的列表打印出来瞧瞧，这里获取到列表中元素序号为1的元素，也就是获取到头衔所在的那部分，即Mr. Owen Harris这部分
'''
#Mr. Owen Harris
str1=name1.split(',')[1]
'''继续对字符串Mr. Owen Harris按分隔符'.'拆分，得到这样一个列表[Mr, Owen Harris]
这里获取到列表中元素序号为0的元素，也就是获取到头衔所在的那部分Mr
'''
str2=str1.split('.')[0]
#strip() 方法用于移除字符串头尾指定的字符（默认为空格）
str3=str2.strip()
'''
定义函数：从姓名中获取头衔
'''
def getTitle(name):
    str1=name.split(',')[1]
    str2=str1.split('.')[0]
    str3=str2.strip()
    return str3
#存放提取后的特征
titleDf = pd.DataFrame()
#map函数：对Series每个数据应用自定义的函数计算
titleDf['Title'] = full['Name'].map(getTitle)
'''
定义以下几种头衔类别：
Officer政府官员
Royalty王室（皇室）
Mr已婚男士
Mrs已婚妇女
Miss年轻未婚女子
Master有技能的人/教师
'''
#姓名中头衔字符串与定义头衔类别的映射关系
title_mapDict = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }
#map函数：对Series每个数据应用自定义的函数计算
titleDf['Title'] = titleDf['Title'].map(title_mapDict)
#使用get_dummies进行one-hot编码
titleDf = pd.get_dummies(titleDf['Title'])
#添加one-hot编码产生的虚拟变量（dummy variables)到泰坦尼克号数据集full
full = pd.concat([full,titleDf],axis=1)
#删掉姓名这一列
full.drop('Name', axis=1, inplace=True)

# print full.head()

#补充知识：匿名函数
'''
python 使用 lambda 来创建匿名函数。
所谓匿名，意即不再使用 def 语句这样标准的形式定义一个函数，如下：
lambda 参数1，参数2：函数体或者表达式
'''
# 定义匿名函数：对两个数相加
sum = lambda a,b: a+b

#调用sum函数
# print("相加后的值为：", sum(10,20))
# 相加后的值为：30
#存放客舱号信息
cabinDf = pd.DataFrame()

'''
客舱号的类别值是首字母，例如：
C85类别映射为首字母C
'''
full['Cabin']=full['Cabin'].map(lambda c : c[0])

##使用get_dummies进行one-hot编码，列名前缀是Cabin
cabinDf = pd.get_dummies(full['Cabin'], prefix = 'Cabin')
#添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full,cabinDf],axis=1)
#删掉客舱号这一列
full.drop('Cabin',axis=1,inplace=True)

# print full.head()

#存放家庭信息
familyDf = pd.DataFrame()
'''
家庭人数=同代直系亲属数（Parch）+不同代直系亲属数（SibSp）+乘客自己
（因为乘客自己也是家庭成员的一个，所以这里加1）
'''
familyDf['FamilySize'] = full['Parch'] + full['SibSp'] + 1

'''
家庭类别：
小家庭Family_Single：家庭人数=1
中等家庭Family_Small: 2<=家庭人数<=4
大家庭Family_Large: 家庭人数>=5
'''
#if 条件为真的时候返回if前面内容，否则返回0
familyDf['Family_Single'] = familyDf['FamilySize'].map(lambda s : 1 if s==1 else 0)
familyDf['Family_Small']  = familyDf['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)
familyDf['Family_Large']  = familyDf['FamilySize'].map(lambda s : 1 if 5<=s else 0)
#添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full,familyDf],axis=1)

# print full.head()

full['IsChild'] = np.where(full['Age']<=12,1,0)
# full['Age'] = pd.cut(full['Age'],5)
# full = full.drop('Age',axis=1)

full['Age'] = pd.qcut(full.Age,5)
ageDf = pd.get_dummies(full.Age,prefix='Age')
full = pd.concat([full,ageDf],axis=1).drop('Age',axis=1)
# print full.head()

# print full.head()
# print full.describe()
# print full.info()

full['Ticket_Lett'] = full['Ticket'].apply(lambda x: str(x)[0])
full['Ticket_Lett'] = full['Ticket_Lett'].apply(lambda x: str(x))

full['High_Survival_Ticket'] = np.where(full['Ticket_Lett'].isin(['1', '2', 'P']),1,0)
full['Low_Survival_Ticket'] = np.where(full['Ticket_Lett'].isin(['A','W','3','7']),1,0)
full = full.drop(['Ticket','Ticket_Lett'],axis=1)

corrDf = full.corr()
'''
查看各个特征与生成情况（Survived）的相关系数，
ascending=False表示按降序排列
'''
# print corrDf['Survived'].sort_values(ascending = False)
# print full.head()

#特征选择

full_X = pd.concat([titleDf,
                    pclassDf,
                    familyDf,
                    fareDf,
                    cabinDf,
                    embarkedDf,
                    full['Sex'],
                    full['High_Survival_Ticket'],
                    full['Low_Survival_Ticket'],
                    nameLenDf,
                    full['IsChild'],
                    ageDf
                   ], axis=1)

'''
full_X = pd.concat([
full['Mr'],
full['Pclass_3'],
full['FamilySize'],
full['Family_Small'],
fareDf,
full['Embarked_S'],
full['Sex'],
full['High_Survival_Ticket'],
full['Low_Survival_Ticket'],
    nameLenDf,
    ageDf,
# full['Master'],
# full['Mrs'],
# full['Pclass_2'],
# full['Cabin_C'],
# full['Cabin_E'],
# full['Cabin_U'],
# full['IsChild'],
# full['Low_Survival_Ticket'],

                   ], axis=1)


full_X = pd.concat([full['Mrs'],
                    full['Miss'],
                    full['High_Survival_Ticket'],
                    nameLenDf,
                    full['Pclass_1'],
                    full['Family_Small'],
                    fareDf,
                    ageDf,
                    full['Cabin_B'],
                    full['Embarked_C'],
                    full['Cabin_D'],
                    full['Cabin_E'],
                    full['Survive_male_family'],
                    full['IsChild'],
                    full['Cabin_C'],
                    full['Pclass_2'],
                    full['Master'],
                    full['Parch'],
                    full['Cabin_F'],
                    full['Royalty'],
                    full['Cabin_A'],
                    full['FamilySize'],
                    full['Cabin_G'],
                    full['Embarked_Q']
                   ], axis=1)
'''
# print full_X.head()
# print full_X.describe()

#原始数据集有891行
sourceRow=891
'''
sourceRow是我们在最开始合并数据前知道的，原始数据集有总共有891条数据
从特征集合full_X中提取原始数据集提取前891行数据时，我们要减去1，因为行号是从0开始的。
'''
#原始数据集：特征
source_X = full_X.loc[0:sourceRow-1,:]
#原始数据集：标签
source_y = full.loc[0:sourceRow-1,'Survived']
#预测数据集：特征
pred_X = full_X.loc[sourceRow:,:]

# print source_X
# print source_y
print source_X.head()

grd = ensemble.RandomForestClassifier(n_estimators=30)
grd.fit(source_X,source_y)
print grd.feature_importances_
# plot_importance(grd)
# plt.show()

from sklearn.model_selection import train_test_split
#建立模型用的训练数据集和测试数据集
train_X, test_X, train_y, test_y = train_test_split(source_X,
                                                   source_y,
                                                   test_size=0.2)
# print train_X
# print train_y

'''
#第1步：导入算法
from sklearn.linear_model import LogisticRegression
#第2步：创建模型：逻辑回归（logistic regression)
model = LogisticRegression()
#第3步：训练模型
model.fit(train_X, train_y)
print model.score(test_X, test_y)
#使用机器学习模型，对预测数据集中的生存情况进行预测
pred_Y = model.predict(pred_X)

# 生成的预测值是浮点数（0.0，1.0）但是kaggle要求提交的结果是整型（0，1）所以要对数据类型进行转换
pred_Y=pred_Y.astype(int)
#乘客id
passenger_id = full.loc[sourceRow:,'PassengerId']
#数据框：乘客id，预测生存情况的值
predDf = pd.DataFrame(
    {'PassengerId': passenger_id,
    'Survived':pred_Y})
predDf.shape
predDf.head()
#保存结果
predDf.to_csv('titanic_pred_new.csv', index = False)
'''

# fit到BaggingRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)

bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False)
bagging_clf.fit(train_X, train_y)
print bagging_clf.score(test_X, test_y)
predictions = bagging_clf.predict(pred_X)

result = pd.DataFrame({'PassengerId':test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv("logistic_regression_bagging_predictions1.csv", index=False)

'''
xgb_clf = XGBClassifier(learning_rate=0.1, max_depth=2, silent=True, objective='binary:logistic')
# xgb_clf = XGBClassifier()
xgb_clf.fit(train_X.values, train_y.values)
print xgb_clf.score(test_X.values, test_y.values)

predictions = xgb_clf.predict(pred_X.values)

result = pd.DataFrame({'PassengerId':test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv("XGBClassifier_predictions2.csv", index=False)


'''

params = {
    'learning_rate':0.08,
    'n_estimators':500,
    'max_depth':3,
    'min_child_weight':1,
    'gamma':0.4,
    'subsample':0.95,
    'colsample_bytree':0.6,
    'reg_alpha': 0.5,
    # 'reg_lambda': 0.0,
    # #在各类别样本十分不平衡时，把这个参数设定为正值，可以是算法更快收敛
    # 'scale_pos_weight':1,
    'objective':'binary:logistic',
    'scale_pos_weight':1,
    'seed':27
}

# clf = XGBClassifier(**params)
# 简单初始化xgb的分类器就可以
clf =XGBClassifier(learning_rate=0.1, max_depth=6, silent=True,
                   n_estimators=32, objective='binary:logistic')

'''
grid_params = {
    # 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
    # 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05],
    'reg_alpha': np.linspace(0, 0.5, 5),
    # 'reg_lambda': np.linspace(0, 0.05, 5),  # 得到最佳参数{'reg_alpha':0,'reg_lambda:0.0125'},Accuracy：96.6%
    # 'subsample': [i / 100.0 for i in range(50, 100, 5)],
    # 'subsample': [i / 100.0 for i in range(75, 90, 5)],
    # 'colsample_bytree': [i / 100.0 for i in range(50, 100, 5)],
    # 'colsample_bytree': [i / 100.0 for i in range(75, 90, 5)],
    # 'subsample':[i/10.0 for i in range(6,10)],
    # 'colsample_bytree':[i/10.0 for i in range(6,10)],
    # 'gamma':[i/10.0 for i in range(0,5)],      #得到最佳参数0，Accuracy：96.5%
    # 'max_depth':list(range(3,15,1)),
    # 'min_child_weight':list(range(1,6,1)),            #得到最佳参数{'max_depth':12,'min_child_weight:1'},Accuracy：96.5%

    # 'n_estimators': list(range(100, 1101, 100)),#得到最佳参数500，Accuracy：96.4%
    # 'learning_rate':np.linspace(0.01,0.2,20)  #得到最佳参数0.01，Accuracy：96.4
}
grid = GridSearchCV(clf,grid_params)
grid.fit(train_X.values,train_y.values)
print(grid.best_params_)
print(grid.best_score_)
'''
# print("Accuracy:{0:.1f}%".format(100*grid.best_score_))

clf.fit(train_X.values, train_y.values)
print clf.score(test_X.values, test_y.values)

predictions = clf.predict(pred_X.values)

result = pd.DataFrame({'PassengerId':test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv("XGBClassifier_predictions6.csv", index=False)
plot_importance(clf)
plt.show()


'''
# 设置boosting迭代计算次数
param_test = {
    'n_estimators': range(30, 50, 2),
    'max_depth': range(2, 7, 1)
}
grid_search = GridSearchCV(estimator = clf, param_grid = param_test, scoring='accuracy', cv=5)
grid_search.fit(train_X.values, train_y.values)
print grid_search.grid_scores_, grid_search.best_params_, grid_search.best_score_


# 简单初始化xgb的分类器就可以
clf =XGBClassifier(learning_rate=0.1, max_depth=2, silent=True, objective='binary:logistic')

# 设置boosting迭代计算次数
param_test = {
    'n_estimators': range(30, 50, 2),
    'max_depth': range(2, 7, 1)
}
grid_search = GridSearchCV(estimator = clf, param_grid = param_test,
scoring='accuracy', cv=5)
grid_search.fit(train_X.values, train_y.values)

'''






