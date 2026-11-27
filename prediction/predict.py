
import argparse
import os

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator


def main():
    parser = argparse.ArgumentParser(
        description="Energy Consumption Prediction"
    )

    parser.add_argument("test_file")

    parser.add_argument(
        "--model",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "model",
            "energy_model"
        )
    )

    args = parser.parse_args()

    spark = SparkSession.builder \
        .appName("CS643-Energy-Prediction") \
        .master("local[2]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    try:
        test_data = spark.read.csv(
            args.test_file,
            header=True,
            inferSchema=True
        )

        model = PipelineModel.load(
            os.path.abspath(args.model)
        )

        predictions = model.transform(test_data)

        evaluator = RegressionEvaluator(
            labelCol="Energy Consumption",
            predictionCol="prediction",
            metricName="rmse"
        )

        rmse = evaluator.evaluate(predictions)

        print("\n===== PREDICTION RESULTS =====")
        print("Test records:", test_data.count())
        print(f"RMSE: {rmse:.6f}")

        predictions.select(
            "Energy Consumption",
            "prediction"
        ).show(10, truncate=False)

        print("PREDICTION_SUCCESS")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
