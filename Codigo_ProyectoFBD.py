from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, unix_timestamp, when, hour, dayofweek

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


#Sesion de Spark
spark = SparkSession.builder \
    .appName("NYC Taxi Tips High-Low Classification") \
    .master("local[*]") \
    .config("spark.driver.memory", "6g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


#Carga de los datos
df = spark.read.csv("yellow_tripdata_2015-01.csv", header=True, inferSchema=True)


#Caracteristicas

df = df.withColumn(
    "trip_duration_min",
    (unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime")) / 60
)

df = df.filter(
    (col("trip_distance") > 0) &
    (col("fare_amount") > 0) &
    (col("tip_amount") >= 0) &
    (col("trip_duration_min") > 0)
)

df = df.withColumn("tip_ratio", col("tip_amount") / col("fare_amount"))

df = df.withColumn(
    "label",
    when(col("tip_ratio") >= 0.20, 1).otherwise(0)
)

df = df.withColumn("pickup_hour", hour("tpep_pickup_datetime"))
df = df.withColumn("pickup_dayofweek", dayofweek("tpep_pickup_datetime"))

#Seleccion de columnas
df = df.select(
    "trip_distance",
    "trip_duration_min",
    "passenger_count",
    "pickup_hour",
    "pickup_dayofweek",
    "label"
)

#Muestra
df = df.sample(withReplacement=False, fraction=0.05, seed=42)


assembler = VectorAssembler(
    inputCols=[
        "trip_distance",
        "trip_duration_min",
        "passenger_count",
        "pickup_hour",
        "pickup_dayofweek"
    ],
    outputCol="features_raw",
    handleInvalid="skip"
)

#Normalizacion
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withMean=False,
    withStd=True
)

# train-test split
train, test = df.randomSplit([0.8, 0.2], seed=42)


#Modelo 1: Logistic Regression
lr = LogisticRegression(
    labelCol="label",
    featuresCol="features"
)

pipeline_lr = Pipeline(stages=[assembler, scaler, lr])

model_lr = pipeline_lr.fit(train)

pred_lr = model_lr.transform(test)


#Modelo 2: Random Forest
rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features_raw",
    numTrees=20,
    maxDepth=6
)

pipeline_rf = Pipeline(stages=[assembler, rf])

model_rf = pipeline_rf.fit(train)

pred_rf = model_rf.transform(test)


#Evaluacion
auc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

#Logistic Regression
auc_lr = auc_eval.evaluate(pred_lr)
f1_lr = f1_eval.evaluate(pred_lr)

print("===== Logistic Regression =====")
print("AUC:", auc_lr)
print("F1 :", f1_lr)

#Random Forest
auc_rf = auc_eval.evaluate(pred_rf)
f1_rf = f1_eval.evaluate(pred_rf)

print("===== Random Forest =====")
print("AUC:", auc_rf)
print("F1 :", f1_rf)


#Guardar metricas en un CSV
metricas = [
    Row(modelo="Logistic Regression", auc=float(auc_lr), f1=float(f1_lr)),
    Row(modelo="Random Forest", auc=float(auc_rf), f1=float(f1_rf))
]

metricas_df = spark.createDataFrame(metricas)

metricas_df.coalesce(1).write.mode("overwrite").csv("metricas_modelos_propina")


#Guardar en un CSV
pred_rf = pred_rf.withColumn("prob_array", vector_to_array("probability"))

pred_rf = pred_rf.withColumn("prob_low",  col("prob_array")[0]) \
                 .withColumn("prob_high", col("prob_array")[1])

predicciones_ejemplo = pred_rf.select(
    "trip_distance",
    "trip_duration_min",
    "passenger_count",
    "pickup_hour",
    "pickup_dayofweek",
    "label",
    "prediction",
    "prob_low",
    "prob_high"
)

predicciones_ejemplo.limit(1000) \
    .coalesce(1) \
    .write.mode("overwrite") \
    .csv("predicciones_propinas_rf")

