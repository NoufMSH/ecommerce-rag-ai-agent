FROM python:3.11-slim-bookworm

# PySpark needs a JRE; procps avoids a known class of Spark-in-minimal-container hangs.
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jre-headless procps && \
    rm -rf /var/lib/apt/lists/*

# Resolve JAVA_HOME dynamically so this works on both amd64 and arm64 base images
# (Debian's install path differs by arch: java-17-openjdk-amd64 vs -arm64).
RUN JAVA_PATH=$(readlink -f $(which java)) && \
    ln -s "$(dirname "$(dirname "$JAVA_PATH")")" /usr/lib/jvm/default-java
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Avoids Spark trying (and failing) to resolve a container hostname.
ENV SPARK_LOCAL_IP=127.0.0.1

WORKDIR /app

# Install CPU-only torch first, from PyTorch's dedicated CPU wheel index. Without
# this, sentence-transformers pulls in the default (CUDA) torch build, which drags
# in several GB of NVIDIA driver packages this CPU-only container will never use.
RUN pip install --no-cache-dir --timeout=120 --retries=5 torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# --retries/--timeout make this resilient to transient network hiccups on large
# wheels like pyspark (this line failed once with a bare ReadTimeoutError).
RUN pip install --no-cache-dir --timeout=120 --retries=5 -r requirements.txt
COPY . .

# Bake the ETL + embedding index build into the image so `docker compose up` starts fast.
RUN python src/etl_pyspark.py && python src/build_index.py

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.headless=true"]
