import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

np.random.seed(0) 

df = pd.DataFrame(np.random.rand(50, 2), columns=['x坐标', 'y坐标'])
df.plot(kind='scatter', x='x坐标', y='y坐标');
plt.show()

plt.figure()
df['x坐标'].hist(color='k', bins=10)
plt.show()

out ,bins= pd.qcut(df['x坐标'], 10,retbins=True)
print(out.value_counts())

