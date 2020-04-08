
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

n_samples = 50000
centers = [(-5, -5), (0, 0), (5, 5)]
X,y=make_blobs(n_samples=n_samples,n_features=2,cluster_std=2.5,centers=centers,shuffle=False, random_state=42)
y[:n_samples // 2] = 0
y[n_samples // 2:] = 1
sample_weight = np.random.RandomState(42).rand(y.shape[0])
X_train, X_test, y_train, y_test,sw_train,sw_test=train_test_split(X, y, sample_weight, test_size=0.9, random_state=42)
clf = GaussianNB()
clf.fit(X_train, y_train)
prob_pos_clf = clf.predict_proba(X_test)[:, 1]
target_pred = clf.predict(X_test)

score = accuracy_score(y_test, target_pred, normalize = True)
print("accuracy score:",score)

clf_score = brier_score_loss(y_test, prob_pos_clf, sw_test)
print("clf_score: %1.3f" % clf_score)

## #############################################################################
## Plot the data and the predicted probabilities
plt.figure()
y_unique = np.unique(y)
patterns = ['^','o']#cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
for this_y, marker in zip(y_unique, patterns):
    this_X = X_train[y_train == this_y]
    this_sw = sw_train[y_train == this_y]
    plt.scatter(this_X[:, 0], this_X[:, 1], s=this_sw * 50, marker=marker,
                alpha=0.5, edgecolor='k',
                label="类别 %s" % this_y)
plt.legend(loc="best")
plt.xlabel("x坐标")
plt.ylabel("y坐标")
plt.title("样本数据")
plt.show()

plt.figure()
order = np.lexsort((prob_pos_clf, ))
plt.plot(prob_pos_clf[order], 'r')
plt.ylim([-0.05, 1.05])
plt.xlabel("样本")
plt.ylabel("P(y=1)")
plt.title("高斯贝叶斯分类")
plt.show()

