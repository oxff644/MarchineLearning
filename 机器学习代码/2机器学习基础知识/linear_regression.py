import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state

n = 16
rs = check_random_state(0)
X = rs.randint(1,100,size=(n,))#np.arange(n)
X.sort()
y = rs.randint(-20, 50, size=(n,)) + 50 * np.log(1 + np.arange(n))
y = [int(yy) for yy in y]
print("X:",X)
print("Y:",y)
model2 = LinearRegression()
model2.fit(X[:, np.newaxis], y)
m = model2.coef_[0]
b = model2.intercept_
print(' y = {0} * x + {1}'.format(m, b))

r2 = model2.score(X[:, np.newaxis], y)
print("r2:",r2)


plt.scatter(X, y,color='g')
plt.plot(X, model2.predict(X[:, np.newaxis]),color='k')
plt.show()
