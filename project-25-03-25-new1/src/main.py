from pyspark.sql import SparkSession
from pyspark.sql.functions import struct
from datetime import datetime
import mlflow.pyfunc
import os

def main():
  spark = SparkSession.builder.appName("batch-inference").getOrCreate()
  
  s3_access_key = os.getenv("AWS_ACCESS_KEY_ID")
  s3_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
  s3_host_port = os.getenv("BATCH_S3_ENDPOINT_URL")
  mlflow_host_port = os.getenv("MLFLOW_INTERNAL_URL")
  s3_input_path = "s3a://data/input-data.parquet"
  s3_output_path = "s3a://data/output-data.parquet"
  
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  output_path_with_timestamp = f"{s3_output_path}_{timestamp}"
  
  if not s3_access_key or not s3_secret_key:
    raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set in the environment variables")
    
  if not s3_host_port:
    raise ValueError("BATCH_S3_ENDPOINT_URL must be set in the environment variables")
  
  if not mlflow_host_port:
    raise ValueError("MLFLOW_INTERNAL_URL must be set in the environment variables")
    
  if not s3_input_path:
    raise ValueError("s3_input_path must be properly configured")
    
  if not s3_output_path:
    raise ValueError("s3_output_path must be properly configured")

  spark.sparkContext._jsc \
      .hadoopConfiguration().set("fs.s3a.access.key", s3_access_key)
  spark.sparkContext._jsc \
      .hadoopConfiguration().set("fs.s3a.secret.key", s3_secret_key)
  spark.sparkContext._jsc \
      .hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
  spark.sparkContext._jsc \
      .hadoopConfiguration().set("fs.s3a.aws.credentials.provider",
                             "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
  spark.sparkContext._jsc \
      .hadoopConfiguration().set("fs.s3a.path.style.access", "true")
  spark.sparkContext._jsc \
      .hadoopConfiguration().set("fs.s3a.endpoint", s3_host_port)

  s3_input_path = "s3a://data/input-data.parquet"
  s3_output_path = "s3a://data/output-data.parquet"
  model_uri = "models:/iris-debug_2/1"
  
  mlflow.set_tracking_uri(mlflow_host_port)

  model_udf = mlflow.pyfunc.spark_udf(spark, model_uri)

  input_df = spark.read.parquet(s3_input_path)

  output_df = input_df.withColumn("prediction", model_udf(struct(*input_df.columns)))
  
  output_df.write.mode("overwrite").parquet(output_path_with_timestamp)

  output_df.show()

  spark.stop()

if __name__ == "__main__":
  main()
