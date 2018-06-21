#coding:utf-8
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score

#read data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

def clean_data(titanic):#填充空数据 和 把string数据转成integer表示
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
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

    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

    return titanic
# 对数据进行清洗
train_data = clean_data(train)
test_data = clean_data(test)

features = ["Pclass", "sex", "child", "fimalysize", "Fare", "embark", "cabin"]

# 简单初始化xgb的分类器就可以
clf =XGBClassifier(learning_rate=0.1, max_depth=6, silent=True,
                   n_estimators=32, objective='binary:logistic')
'''
# 设置boosting迭代计算次数
param_test = {
    'n_estimators': range(30, 50, 2),
    'max_depth': range(2, 7, 1)
}
grid_search = GridSearchCV(estimator = clf, param_grid = param_test, scoring='accuracy', cv=5)
grid_search.fit(train_data[features], train_data["Survived"])
# grid_search.grid_scores_, grid_search.best_params_, grid_search.best_score_

print(grid_search.best_params_)
print(grid_search.best_score_)
'''
clf.fit(train_data[features].values, train_data["Survived"].values)
# print clf.score(test_X.values, test_y.values)

predictions = clf.predict(test_data[features].values)

result = pd.DataFrame({'PassengerId':test_data['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv("XGBClassifier_predictions5.csv", index=False)




