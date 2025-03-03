from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
import mlflow.pyfunc

INPUT_PARQUET_PATH = "data/input_data.parquet" # Входные данные
OUTPUT_PARQUET_PATH = "data/output_data.parquet" # Выходные предсказания
MLFLOW_MODEL_URI = "models:/my_model/Production" # MLflow-модель
FEAST_REPO_PATH = "feast_repo/"  # Путь к репозиторию Feast
FEATURE_SERVICE_NAME = "customer_features"  # Название Feature Service

spark = SparkSession.builder \
  .appName("pd-model-batch-inference") \
  .getOrCreate()

model_udf = mlflow.pyfunc.spark_udf(spark, model_uri=MLFLOW_MODEL_URI)

df = spark.read.parquet(INPUT_PARQUET_PATH)

df.printSchema()
df.show(5)

df.createOrReplaceTempView("input_data")
df.printSchema()
df.show(5)

entity_df = spark.sql("""SELECT DISTINCT entity_id FROM input_data WHERE event_timestamp >= DATE_SUB(current_date(), 30)""")
fs = FeatureStore(repo_path=FEAST_REPO_PATH)
feature_service = fs.get_feature_service(FEATURE_SERVICE_NAME)
feature_vector = fs.get_offline_features(
  entity_df=entity_df
  features=feature_service,
).to_df()

df_pred = df.withColumn("pred", model_udf(feature_vector))
df_pred.write.mode("overwrite").parquet(OUTPUT_PARQUET_PATH)

spark.stop()
