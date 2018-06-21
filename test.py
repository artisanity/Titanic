# coding: utf-8

import xgboost
from xgboost.sklearn import XGBClassifier

'''
import numpy as np
import pandas as pd
from collections import OrderedDict
import pandas as pd

examDict={
    '学习时间':[0.50,0.75,1.00,1.25,1.50,1.75,1.75,2.00,2.25,2.50,2.75,3.00,3.25,3.50,4.00,4.25,4.50,4.75,5.00,5.50],
    '分数':    [10,  22,  13,  43,  20,  22,  33,  50,  62,  48,  55,  75,  62,  73,  81,  76,  64,  82,  90,  93]
}
examOrderDict=OrderedDict(examDict)
examDf=pd.DataFrame(examOrderDict)

#提取特征和标签
#特征features
exam_X=examDf.loc[:,'学习时间']
#标签labes
exam_y=examDf.loc[:,'分数']

from sklearn.model_selection import train_test_split
#X_train:训练数据标签, X_test：测试数据标签, y_train：训练数据特征, y_test：测试数据特征,
#exam_X：样本特征, exam_y：样本标签, train_size：训练数据占比
X_train, X_test, y_train, y_test = train_test_split(exam_X, exam_y, test_size=0.3)
# X_train, X_test, y_train, y_test = train_test_split(exam_X, exam_y, train_size = .8)

# #第1步：导入线性回归
# from sklearn.linear_model import LinearRegression
# #第2步：创建模型：线性回归
# model = LinearRegression()
# #第3步：训练模型
# model.fit(X_train , y_train)

#将训练数据特征转换成二维数组X行*1列
X_train=X_train.values.reshape(-1,1)
#将测试数据特征转换成二维数组X行*1列
X_test=X_test.values.reshape(-1,1)
#第1步：导入线性回归
from sklearn.linear_model import LinearRegression
#第2步：创建模型：线性回归
model = LinearRegression()
#第3步：训练模型
model.fit(X_train , y_train)

#截距
a=model.intercept_
#回归系数
b=model.coef_
print('最佳拟合线：截距a=',a,',回归系数b=',b)

#绘图
import matplotlib.pyplot as plt
#训练数据散点图
plt.scatter(X_train, y_train, color='blue', label="train data")
#训练数据的预测值
y_train_pred = model.predict(X_train)
#绘制最佳拟合线
plt.plot(X_train, y_train_pred, color='black', linewidth=3, label="best line")
#添加图标标签
plt.legend(loc=2)
plt.xlabel("Hours")
plt.ylabel("Score")
#显示图像
plt.show()

#评估模型：决定系数R平方
print model.score(X_test, y_test)

'''



'''
lr = LogisticRegression()
svc = SVC()
knn = KNeighborsClassifier(n_neighbors = 3)
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=300,min_samples_leaf=4,class_weight={0:0.745,1:0.255})
gbdt = GradientBoostingClassifier(n_estimators=500,learning_rate=0.03,max_depth=3)
# xgb = XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)

# clfs = [lr, svc, knn, dt, rf, gbdt, xgb]
clfs = [lr, svc, knn, dt, rf, gbdt]


kfold = 10
cv_results = []
for classifier in clfs :
    cv_results.append(cross_val_score(classifier, train_X, y = train_y, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,
                       "Algorithm":["LR","SVC",'KNN','decision_tree',"random_forest","GBDT","xgbGBDT"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
'''

'''
class Ensemble(object):

    def __init__(self, estimators):
        self.estimator_names = []
        self.estimators = []
        for i in estimators:
            self.estimator_names.append(i[0])
            self.estimators.append(i[1])
        self.clf = LogisticRegression()

    def fit(self, train_x, train_y):
        for i in self.estimators:
            i.fit(train_x, train_y)
        x = np.array([i.predict(train_x) for i in self.estimators]).T
        y = train_y
        self.clf.fit(x, y)

    def predict(self, x):
        x = np.array([i.predict(x) for i in self.estimators]).T
        # print(x)
        return self.clf.predict(x)

    def score(self, x, y):
        s = precision_score(y, self.predict(x))
        return s

bag = Ensemble([('lr',lr),('rf',rf),('svc',svc),('gbdt',gbdt)])
# bag = Ensemble([('xgb',xgb),('lr',lr),('rf',rf),('svc',svc),('gbdt',gbdt)])
score = 0
for i in range(0,10):
    num_test = 0.20
    X_train, X_cv, Y_train, Y_cv = train_test_split(train_X, train_y, test_size=num_test)
    bag.fit(X_train, Y_train)
    #Y_test = bag.predict(X_test)
    acc_xgb = round(bag.score(X_cv, Y_cv) * 100, 2)
    score+=acc_xgb
print(score/10)  #0.8786
'''