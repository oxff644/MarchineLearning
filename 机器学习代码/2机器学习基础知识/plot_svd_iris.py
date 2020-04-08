from numpy import *
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import *

df = pd.read_csv('iris.data.csv', header=None)
df[4] = df[4].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})
print(df.head())
data = df.iloc[:, :-1].values
samples,features = df.shape
S=2

U, s, V = linalg.svd( data )
Sig = mat(eye(S)*s[:S])
print(Sig)
newdata = U[:,:S]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
marks = ['o','^','+']
for i in range(samples):
    ax.scatter(newdata[i,0],newdata[i,1],c='black',marker=marks[int(data[i,-1])])
plt.xlabel('SVD1')
plt.ylabel('SVD2')
plt.show()



