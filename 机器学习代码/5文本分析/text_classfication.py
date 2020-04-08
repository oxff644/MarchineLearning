# coding: utf-8
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

sc =SparkContext()
sqlContext = SQLContext(sc)
data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('train.csv')
data.show(10)
#正则化分词
regexTokenizer = RegexTokenizer(inputCol="Detail", outputCol="words", pattern="\\W")
add_stopwords = ["this","that","amp","rt","t","c","the","me","he","it","a","an","is","has","had"]
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)
label_stringIdx = StringIndexer(inputCol = "Category", outputCol = "label")

#TF-IDF提取特征
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=100000)
idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=3)

pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, hashingTF, idf, label_stringIdx])
pipelineFit = pipeline.fit(data)
dataset = pipelineFit.transform(data)
dataset.select("Category","filtered","features","label").show(10)
dataset.show(10)

(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 42)
print("训练集数量: " + str(trainingData.count()))
print("测试集数量: " + str(testData.count()))

#训练逻辑回归模型
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = lr.fit(trainingData)
predictions = lrModel.transform(testData)
predictions.select("Detail","Category","probability","label","prediction").show(n = 10, truncate = 30)

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
print("accuracy:",evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"}))

