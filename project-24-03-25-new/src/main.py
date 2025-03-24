from pyspark.sql import SparkSession
from pyspark.sql.functions import struct
import mlflow.pyfunc

def main():
  spark = SparkSession.builder.appName("appName").getOrCreate()

  spark.sparkContext._jsc \
      .hadoopConfiguration().set("fs.s3a.access.key", "1iQ1gBmOkEiB7S96bpzU")
  spark.sparkContext._jsc \
      .hadoopConfiguration().set("fs.s3a.secret.key", "OOV0ZHaqFgwCacz5EgK4f00k6uQjYgUH4ravZgNV")
  spark.sparkContext._jsc \
      .hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
  spark.sparkContext._jsc \
      .hadoopConfiguration().set("fs.s3a.aws.credentials.provider",
                             "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
  spark.sparkContext._jsc \
      .hadoopConfiguration().set("fs.s3a.path.style.access",
                             "true")
  spark.sparkContext._jsc \
      .hadoopConfiguration().set("fs.s3a.endpoint",
                             "http://10.96.46.93:80")


  s3_input_path = "s3a://data/input-data.parquet"
  s3_output_path = "s3a://data/output-data.parquet"
  model_uri = "models:/iris-debug_2/1"
  mlflow.set_tracking_uri("http://10.96.166.129:80")

  model_udf = mlflow.pyfunc.spark_udf(spark, model_uri)

  input_df = spark.read.parquet(s3_input_path)

  output_df = input_df.withColumn("prediction", model_udf(struct(*input_df.columns)))

  output_df.write.mode("overwrite").parquet(s3_output_path)
  output_df.show()

  spark.stop()

if __name__ == "__main__":
  main()
