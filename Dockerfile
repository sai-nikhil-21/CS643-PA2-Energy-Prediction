FROM python:3.11-slim-bookworm

RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jre-headless procps && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pyspark==3.5.3 \
    numpy==1.26.4

ENV PYSPARK_PYTHON=/usr/local/bin/python
ENV PYSPARK_DRIVER_PYTHON=/usr/local/bin/python
ENV SPARK_LOCAL_IP=127.0.0.1

WORKDIR /app

COPY prediction/predict.py /app/prediction/predict.py
COPY model/energy_model /app/model/energy_model

ENTRYPOINT ["spark-submit", "--master", "local[2]", "--driver-memory", "1g", "/app/prediction/predict.py"]
