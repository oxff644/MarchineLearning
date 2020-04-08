from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot

X, y = make_blobs(n_samples=1000, n_features=6, centers=50,
    random_state=0)

pyplot.scatter(X[:, 0], X[:, 1], c=y)
pyplot.show()

clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,
    random_state=0)
scores = cross_val_score(clf, X, y)
print("DecisionTreeClassifier result:",scores.mean())


clf = RandomForestClassifier(n_estimators=10, max_depth=None,
    min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y)
print("RandomForestClassifier result:",scores.mean())


clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,
    min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y)
print("ExtraTreesClassifier result:",scores.mean())

