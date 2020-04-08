import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#——————————————————————————————————————————————————————————
labels = '财经15%', '社会30%', '体育15%','科技10%', '其它30%'
sizes = [15, 30, 15, 10, 30]
explode = (0, 0.1, 0, 0,0)#突出第2项
fig1, ax1 = plt.subplots()
pie = ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=False, startangle=90)
patches = pie[0]
patches[0].set_hatch('.')
patches[1].set_hatch('-')
patches[2].set_hatch('+')
patches[3].set_hatch('x')
patches[4].set_hatch('o')
plt.legend(patches, labels)
ax1.axis('equal')
plt.title('新闻网站用户兴趣分析')
plt.show()
#——————————————————————————————————————————————————————————


#——————————————————————————————————————————————————————————
import numpy as np
N = 5
inMeans = (20, 25, 30, 35, 27)
outMeans = (25, 35, 34, 20, 25)
inStd = (2, 3, 4, 1, 2)
outStd = (3, 5, 2, 3, 3)
ind = np.arange(N)    #Bar坐标位置
width = 0.5     #Bar的宽度

p1 = plt.bar(ind, inMeans, width, yerr=inStd)
p2 = plt.bar(ind, outMeans, width,bottom=inMeans, yerr=outStd)

plt.ylabel('分值')
plt.title('不同组用户下国内外用户分值')
plt.xticks(ind, ('组1', '组2', '组3', '组4', '组5'))
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('国内', '国外'))
plt.show()
#——————————————————————————————————————————————————————————




#——————————————————————————————————————————————————————————
import matplotlib.pyplot as plt
import squarify

squarify.plot(sizes=[20,10,30,40], label=["组A(20%)", "组B(10%)", "组C(30%)", "组D(40%)"], color=["red","green","blue", "grey"], alpha=.4 )
plt.axis('off')
plt.title('不同组用户比例')
plt.show()
#——————————————————————————————————————————————————————————




#——————————————————————————————————————————————————————————
# library
import numpy as np
import seaborn as sns

x=range(21,26)
y=[ [10,4,6,5,3], [12,2,7,10,1], [8,18,5,7,6],[1,8,3,5,9] ]
labels = ['组A','组B','组C','组D']

pal = sns.color_palette("Set1")
plt.stackplot(x,y, labels=labels, colors=pal, alpha=0.7 )
plt.ylabel('分值')
plt.xlabel('年龄')
plt.title('不同组用户区间分值比较')

plt.legend(loc='upper right')
plt.show()
#——————————————————————————————————————————————————————————


