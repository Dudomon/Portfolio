FROM flink:1.18-scala_2.12-java11

# Install Python for PyFlink
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-dev && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install PyFlink and dependencies
RUN pip3 install --no-cache-dir \
    apache-flink==1.18.0 \
    kafka-python==2.0.2 \
    clickhouse-driver==0.2.6 \
    redis==5.0.1

# Copy Flink jobs
COPY jobs/ /opt/flink/jobs/

# Set working directory
WORKDIR /opt/flink

# Expose Flink ports
EXPOSE 8081 6123
