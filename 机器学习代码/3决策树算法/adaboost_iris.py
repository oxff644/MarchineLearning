from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

dataset_all = datasets.load_breast_cancer()
X = dataset_all.data
Y = dataset_all.target

seed = 42
kfold = KFold(n_splits=10, random_state=seed)
dtree = DecisionTreeClassifier(criterion='gini',max_depth=3)
dtree = dtree.fit(X, Y)
result = cross_val_score(dtree, X, Y, cv=kfold)
print("决策树结果：",result.mean())
model = AdaBoostClassifier(base_estimator=dtree, n_estimators=100,random_state=seed)
result = cross_val_score(model, X, Y, cv=kfold)
print("提升法改进结果：",result.mean())
