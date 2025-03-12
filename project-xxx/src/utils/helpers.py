import sys
from random import random
from operator import add  
from pyspark.sql import SparkSession

def helper(): pass

  if __name__ == "__main__":
"""
    Usage: pi [partitions]
"""
 spark = SparkSession\
  .builder\
  .appName("PythonPi")\
  .getOrCreate()
