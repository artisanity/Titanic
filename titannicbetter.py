#coding:utf-8
import numpy as np
import pandas as pd
from sklearn import linear_model
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
# from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
#可视化处理
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score

# 解决图标中文乱码问题
# matplotlib.use('qt4agg')
#指定默认字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'
#解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False

#read data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values

    print known_age
    print unknown_age

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges

    return df


def clean_train_data(titanic):#填充空数据 和 把string数据转成integer表示
    # titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = titanic[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values

    # print known_age
    # print unknown_age

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    titanic.loc[(titanic.Age.isnull()), 'Age'] = predictedAges

    # child
    titanic["child"] = titanic["Age"].apply(lambda x: 1 if x < 15 else 0)

    # sex
    titanic["sex"] = titanic["Sex"].apply(lambda x: 1 if x == "male" else 0)

    titanic["Embarked"] = titanic["Embarked"].fillna("S")
    # embark
    def getEmbark(Embarked):
        if Embarked == "S":
            return 1
        elif Embarked == "C":
            return 2
        else:
            return 3
    titanic["embark"] = titanic["Embarked"].apply(getEmbark)

    # familysize
    titanic["fimalysize"] = titanic["SibSp"] + titanic["Parch"] + 1

    # cabin
    def getCabin(cabin):
        if cabin == "N":
            return 0
        else:
            return 1
    titanic["cabin"] = titanic["Cabin"].apply(getCabin)

    # name
    def getName(name):
        if "Mr" in str(name):
            return 1
        elif "Mrs" in str(name):
            return 2
        else:
            return 0
    titanic["name"] = titanic["Name"].apply(getName)

    # titanic["Fare"] = titanic["Fare"].fillna(8.05)
    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

    return titanic, rfr

def clean_test_data(titanic):#填充空数据 和 把string数据转成integer表示
    # child
    titanic["child"] = titanic["Age"].apply(lambda x: 1 if x < 15 else 0)

    # sex
    titanic["sex"] = titanic["Sex"].apply(lambda x: 1 if x == "male" else 0)

    titanic["Embarked"] = titanic["Embarked"].fillna("S")
    # embark
    def getEmbark(Embarked):
        if Embarked == "S":
            return 1
        elif Embarked == "C":
            return 2
        else:
            return 3
    titanic["embark"] = titanic["Embarked"].apply(getEmbark)

    # familysize
    titanic["fimalysize"] = titanic["SibSp"] + titanic["Parch"] + 1

    # cabin
    def getCabin(cabin):
        if cabin == "N":
            return 0
        else:
            return 1
    titanic["cabin"] = titanic["Cabin"].apply(getCabin)

    # name
    def getName(name):
        if "Mr" in str(name):
            return 1
        elif "Mrs" in str(name):
            return 2
        else:
            return 0
    titanic["name"] = titanic["Name"].apply(getName)

    # titanic["Fare"] = titanic["Fare"].fillna(8.05)
    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

    return titanic

# 对数据进行清洗
train_data,rfr = clean_train_data(train)

test["Fare"] = test["Fare"].fillna(test["Fare"].median())

test.loc[ (test.Fare.isnull()), 'Fare' ] = 0
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[test.Age.isnull()].values
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
test.loc[ (test.Age.isnull()), 'Age' ] = predictedAges

test_data = clean_test_data(test)

print train_data.head()
print train_data.describe()
print train_data.info()
print test_data.head()
print test_data.describe()
print test_data.info()

print train_data['Age'].values
print test_data['Age'].values

# trainplt = train_data['Age'].ceil()
#
# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数
# trainplt.Age.value_counts(ascending=True).plot(kind="bar")
# plt.ylabel(u"年龄")
# plt.title(u"乘客年龄分布")
# plt.show()

features = ["Pclass", "sex", "child", "fimalysize", "Fare", "embark", "cabin"]

'''
# ### （2）选择机器学习算法

# 选择一个机器学习算法，用于模型训练，这里选择逻辑回归（logisic regression）

# 第1步：导入算法
# from sklearn.linear_model import LogisticRegression
# 第2步：创建模型：逻辑回归（logisic regression）
# model = LogisticRegression()

# 随机森林Random Forests Model
# from sklearn.ensemble import RandomForestClassifier, BaggingRegressor
# model = RandomForestClassifier(n_estimators=100)
#0.7765

# 支持向量机Support Vector Machines
# from sklearn.svm import SVC,LinearSVC
# model = SVC()
#0.78

# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingRegressor, RandomForestRegressor

model = GradientBoostingClassifier()
#0.81

#K-nearest neighbors
# from sklearn.neighbors import KNeighborsClassifier
# model = KNeighborsClassifier(n_neighbors = 3)
#0.75

# 朴素贝叶斯Gaussian Naive Bayes
# from sklearn.naive_bayes import GaussianNB
# model = GaussianNB()
#0.81


# ### （3）训练模型

# 第3步：训练模型
# model.fit(train_X,train_y)
model.fit(train_data[features].values, train_data["Survived"].values)

# ### （4）评估模型

# 分类问题，score得到的是模型正确率
# model.score(test_X,test_y)

# ## 5、实施方案

# 使用预测数据集进行预测结果，并保存到csv文件中，最后上传到Kaggle中

# 使用机器学习模型，对预测数据集中的生存情况进行预测
predictions = model.predict(test_data[features].values)
print model
# 保存结果
result = pd.DataFrame({'PassengerId':test_data['PassengerId'].values, 'Survived':predictions.astype(np.int32)})

# result.to_csv("LogisticRegressionPrediction.csv", index=False)
# result.to_csv("RandomForestClassifierPrediction.csv", index=False)
# result.to_csv("SVCPrediction.csv", index=False)
result.to_csv("GradientBoostingClassifierPrediction.csv", index=False)
# result.to_csv("KNeighborsClassifierPrediction.csv", index=False)
# result.to_csv("GaussianNBPrediction.csv", index=False)
'''


# 简单初始化xgb的分类器就可以
clf =XGBClassifier(learning_rate=0.1, max_depth=6, n_estimators=32, silent=True, objective='binary:logistic')

# 设置boosting迭代计算次数
# param_test = {
#     'n_estimators': range(30, 50, 2),
#     'max_depth': range(2, 7, 1)
# }
# grid_search = GridSearchCV(estimator = clf, param_grid = param_test, scoring='accuracy', cv=5)
# grid_search.fit(train_data[features], train_data["Survived"])
# print grid_search.grid_scores_, grid_search.best_params_, grid_search.best_score_

clf.fit(train_data[features].values, train_data["Survived"].values)
# print clf.score(train_data[features].values, train_data["Survived"].values)

predictions = clf.predict(test_data[features].values)

result = pd.DataFrame({'PassengerId':test_data['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv("XGBClassifierPredictionLast.csv", index=False)


'''

# fit到BaggingRegressor之中
# model = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
bagging_clf = BaggingRegressor(model, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False)
bagging_clf.fit(train_data[features].values, train_data["Survived"].values)
print bagging_clf

predictions = bagging_clf.predict(test_data[features].values)

result = pd.DataFrame({'PassengerId':test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv("GradientBoostingClassifierBaggingPrediction.csv", index=False)

'''

