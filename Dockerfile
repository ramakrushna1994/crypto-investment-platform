FROM apache/airflow:3.1.4

USER root

RUN apt-get update && \
    apt-get install -y openjdk-17-jdk libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libjpeg-dev zlib1g-dev build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/airflow/logs/spark && chown -R 50000:50000 /opt/airflow/logs/spark


ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:/home/airflow/.local/bin:$PATH

USER airflow
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

USER airflow
