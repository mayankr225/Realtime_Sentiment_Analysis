FROM flink:1.17.2

USER root

# Install OpenJDK and Python
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk python3 python3-pip && \
    java_path=$(readlink -f /usr/lib/jvm/java-11-openjdk-*) && \
    mkdir -p /opt/java/openjdk && \
    cp -r ${java_path}/* /opt/java/openjdk/ && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    echo "Java path: ${java_path}"

# Set environment variables
ENV JAVA_HOME=/opt/java/openjdk
ENV KAFKA_VERSION=3.1.0
ENV PYFLINK_VERSION=1.17.2
ENV PYTHONPATH=/opt/flink/opt/python
ENV NLTK_DATA=/opt/flink/nltk_data

# Install Python packages
RUN pip3 install \
    apache-flink==${PYFLINK_VERSION} \
    kafka-python \
    confluent-kafka \
    nltk \
    gensim \
    torch \
    numpy \
    protobuf \
    emoji


# Add Kafka connector JARs and set correct permissions
ADD https://repo1.maven.org/maven2/org/apache/flink/flink-connector-kafka/${PYFLINK_VERSION}/flink-connector-kafka-${PYFLINK_VERSION}.jar /opt/flink/lib/
ADD https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/3.5.1/kafka-clients-3.5.1.jar /opt/flink/lib/

COPY nltk_data /opt/flink/nltk_data/

# Fix permissions for Kafka JARs and NLTK data
RUN chown -R flink:flink /opt/flink/lib/flink-connector-kafka-${PYFLINK_VERSION}.jar \
    /opt/flink/lib/kafka-clients-3.5.1.jar \
    /opt/flink/nltk_data && \
    chmod 644 /opt/flink/lib/flink-connector-kafka-${PYFLINK_VERSION}.jar \
    /opt/flink/lib/kafka-clients-3.5.1.jar

# Verify installation
RUN python3 -c "from pyflink.datastream import StreamExecutionEnvironment; print('PyFlink installation successful!')"

USER flink
