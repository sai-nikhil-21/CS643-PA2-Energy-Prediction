
import csv
import os
import socket
import time

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType
)
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler
)
from pyspark.ml.regression import (
    LinearRegression, RandomForestRegressor, GBTRegressor
)
from pyspark.ml.evaluation import RegressionEvaluator


# Connect to the existing Spark cluster (or start a local session).
spark = SparkSession.builder \
    .appName("CS643-Energy-Consumption-Training") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

print("Spark Master:", sc.master)

# Project root on the machine running this script.
BASE_DIR = os.path.expanduser("~/CS643_PA2")

# Explicit schema so numeric columns are doubles and headers match the CSV.
schema = StructType([
    StructField("Building Type", StringType()),
    StructField("Square Footage", DoubleType()),
    StructField("Number of Occupants", DoubleType()),
    StructField("Appliances Used", DoubleType()),
    StructField("Average Temperature", DoubleType()),
    StructField("Day of Week", StringType()),
    StructField("Energy Consumption", DoubleType())
])


def load_dataset(filename):
    """Read a local CSV on the driver, then distribute it across the cluster."""
    path = os.path.join(BASE_DIR, "data", filename)

    # utf-8-sig strips a BOM if the file was saved from Excel.
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        rows = [
            (
                row["Building Type"],
                float(row["Square Footage"]),
                float(row["Number of Occupants"]),
                float(row["Appliances Used"]),
                float(row["Average Temperature"]),
                row["Day of Week"],
                float(row["Energy Consumption"])
            )
            for row in reader
        ]

    # Split the rows into 16 partitions so later stages run in parallel.
    distributed_rows = sc.parallelize(rows, 16)

    # Cache because each model fit and the count() below reuse this data.
    return spark.createDataFrame(
        distributed_rows, schema
    ).cache()


training = load_dataset("TrainingDataset.csv")
validation = load_dataset("ValidationDataset.csv")

print("Training records:", training.count())
print("Validation records:", validation.count())
print("Training partitions:", training.rdd.getNumPartitions())


# Confirm work actually lands on worker hosts, not only the driver.
def check_worker(partition):
    time.sleep(2)
    yield socket.gethostname()


# 32 partitions so several tasks run at once and report distinct hostnames.
hosts = sc.parallelize(range(32), 32) \
    .mapPartitions(check_worker) \
    .distinct().collect()

print("Executor hosts:", hosts)
print("Number of executor hosts:", len(hosts))


# Turn the two categorical columns into numeric features.
# handleInvalid="keep" puts unseen labels into an extra category at prediction time.
building_indexer = StringIndexer(
    inputCol="Building Type",
    outputCol="building_index",
    handleInvalid="keep"
)

day_indexer = StringIndexer(
    inputCol="Day of Week",
    outputCol="day_index",
    handleInvalid="keep"
)

# One-hot encode so building type and weekday are not treated as ordered numbers.
encoder = OneHotEncoder(
    inputCols=["building_index", "day_index"],
    outputCols=["building_encoded", "day_encoded"]
)

# Single feature vector required by the regressors. Label stays separate.
assembler = VectorAssembler(
    inputCols=[
        "building_encoded",
        "Square Footage",
        "Number of Occupants",
        "Appliances Used",
        "Average Temperature",
        "day_encoded"
    ],
    outputCol="features"
)


# Same feature pipeline, three regressors. RMSE on validation picks the winner.
models = {
    "Linear Regression": LinearRegression(
        featuresCol="features",
        labelCol="Energy Consumption",
        maxIter=100,
        regParam=0.0
    ),

    "Random Forest": RandomForestRegressor(
        featuresCol="features",
        labelCol="Energy Consumption",
        numTrees=50,
        maxDepth=5,
        seed=42
    ),

    "Gradient Boosted Trees": GBTRegressor(
        featuresCol="features",
        labelCol="Energy Consumption",
        maxIter=50,
        maxDepth=4,
        seed=42
    )
}

# Root mean squared error: lower means predictions are closer to actual consumption.
evaluator = RegressionEvaluator(
    labelCol="Energy Consumption",
    predictionCol="prediction",
    metricName="rmse"
)

best_model = None
best_name = None
best_rmse = float("inf")

print("\n===== MODEL TRAINING =====")

for name, algorithm in models.items():

    print("\nTraining:", name)

    # Indexers, encoder, and assembler are refit inside each pipeline.
    pipeline = Pipeline(stages=[
        building_indexer,
        day_indexer,
        encoder,
        assembler,
        algorithm
    ])

    model = pipeline.fit(training)

    # Score the held-out validation set; training rows are not used here.
    predictions = model.transform(validation)

    rmse = evaluator.evaluate(predictions)

    print(f"{name} Validation RMSE: {rmse:.6f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_name = name


print("\n===== FINAL RESULTS =====")
print("Best Model:", best_name)
print(f"Best Validation RMSE: {best_rmse:.6f}")

model_path = os.path.join(
    BASE_DIR, "shared", "energy_model"
)

# file:// forces a local filesystem write instead of the default distributed store.
best_model.write().overwrite().save(
    "file://" + model_path
)

print("Model saved to:", model_path)

# Drop cached DataFrames before shutting the session down.
training.unpersist()
validation.unpersist()

spark.stop()
