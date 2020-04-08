import pandas as pd
import xgboost as xgb

df = pd.DataFrame({'x':[1,2,3], 'y':[10,20,30]})
X_train = df.drop('y',axis=1)
Y_train = df['y']
T_train_xgb = xgb.DMatrix(X_train, Y_train)

params = {"objective": "reg:linear","eval_metric":"rmse", "booster":"gblinear"}
gbm = xgb.train(dtrain=T_train_xgb,params=params)
y_test = xgb.DMatrix(pd.DataFrame({'x':[4,5]}))
Y_pred = gbm.predict(y_test)
print(Y_pred)
