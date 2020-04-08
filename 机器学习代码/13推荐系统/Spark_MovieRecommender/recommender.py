import os
import math
import time
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row, SparkSession
from pyspark.mllib.recommendation import ALS

sc = SparkContext()

#文件访问
small_raw_data = sc.textFile('ratings.csv')
small_data = small_raw_data.map(lambda line: line.split(",")).map(lambda col: (col[0], col[1], col[2])).cache()


#按照6:2:2分为训练集、验证集、测试集
training_RDD, validation_RDD, test_RDD = small_data.randomSplit([6, 2, 2], seed=10)
validation_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

#ALS参数配置
seed = 5
iterations = 10
regularization_param = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.02

#模型训练确认rank值（最小误差）
min_error = float('inf')
best_rank = -1
best_iteration = -1
for rank in ranks:
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations, lambda_=regularization_param)
    predict = model.predictAll(validation_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_predictions = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predict)
    error = math.sqrt(rates_predictions.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())
    errors[err] = error
    err += 1
    if error < min_error:
        min_error = error
        best_rank = rank

#以最佳rank值新重训练模型
model = ALS.train(training_RDD, best_rank, seed=seed, iterations=iterations, lambda_=regularization_param)
#模型测试
predictions = model.predictAll(test_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))

rates_and_predictions = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
#计算RMSE指标
error = math.sqrt(rates_predictions.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())
print('Model RMSE = %s' % error)

#预测某一用户对某一电影的评分
user_id = 16
movie_id = 48
predictedRating = model.predict(user_id, movie_id)
print("用户编号:"+str(user_id)+" 对电影:"+str(movie_id)+" 的评分为:"+str(predictedRating))

#向某一用户推荐10部电影
topKRecs = model.recommendProducts(user_id, 10)
print("向用户编号:"+str(user_id)+"的用户推荐10部电影:")
for rec in topKRecs:
    print(rec)

