# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import cross_validation

y_test = [1,1,1,1,1,1,0,1,1,0,1,1,1,1,0,1,1,0,1,0]
y_score =[0.98,0.96,0.92,0.88,0.85,0.83,0.82,0.8,0.78,0.71,0.68,0.64,0.59,0.55,0.52,0.51,0.5,0.48,0.42,0.2]
fpr,tpr,threshold = roc_curve(y_test, y_score)
roc_auc = auc(fpr,tpr)
plt.figure(figsize=(7,7))
plt.plot(fpr, tpr, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % roc_auc) #假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
