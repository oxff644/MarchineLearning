import numpy as np
import random
from sklearn import tree
from graphviz import Source

np.random.seed(42)
X=np.random.randint(10, size=(100, 4))
Y=np.random.randint(2, size=100)
a=np.column_stack((Y,X))
clf = tree.DecisionTreeClassifier(criterion='gini',max_depth=3)
clf = clf.fit(X, Y)
graph = Source(tree.export_graphviz(clf, out_file=None))
graph.format = 'png'
graph.render('cart_tree',view=True)

