import zstandard
import json
from kafka import KafkaProducer
import time
import os
import traceback
import io
import subprocess

KAFKA_BROKER = 'localhost:9092' 
KAFKA_TOPIC = 'reddit_comments'
KAFKA_TOPIC_PROCESSED = 'reddit_comments_processed'
DATA_FILE = 'data/RC_2019-04.zst'
START_TIMESTAMP = 1554076800
END_TIMESTAMP = 1555472130
BATCH_SEND_SPEED_FACTOR = 100

producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BROKER],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def create_kafka_topic(topic_name, bootstrap_servers, partitions=1, replication_factor=1):
    command = [
        'docker-compose', 'exec', 'kafka', 'kafka-topics',
        '--create',
        '--topic', topic_name,
        '--bootstrap-server', bootstrap_servers,
        '--partitions', str(partitions),
        '--replication-factor', str(replication_factor)
    ]
    print(f"Attempting to create Kafka topic: {topic_name}")
    try:
        # Check if the topic already exists to avoid errors
        check_command = [
            'docker-compose', 'exec', 'kafka', 'kafka-topics',
            '--list',
            '--bootstrap-server', bootstrap_servers
        ]
        result = subprocess.run(check_command, capture_output=True, text=True, check=False)
        if topic_name in result.stdout:
            print(f"Topic '{topic_name}' already exists. Skipping creation.")
            return

        # If not, create it
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"Successfully created topic '{topic_name}':")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error creating topic '{topic_name}':")
        print(e.stderr)
    except FileNotFoundError:
        print("Error: 'docker-compose' command not found. Make sure Docker Desktop is running and docker-compose is in your PATH.")
    except Exception as e:
        print(f"An unexpected error occurred during topic creation: {e}")


def process_and_send_comments():
    print(f"Starting data provider for {DATA_FILE}...")
    current_time_offset = 0
    last_timestamp_sent = None

    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at {DATA_FILE}")
        return

    try:
        with open(DATA_FILE, 'rb') as f:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                for line in text_stream:
                    try:
                        comment = json.loads(line)
                        created_utc = comment.get('created_utc')

                        if created_utc and START_TIMESTAMP <= created_utc <= END_TIMESTAMP:
                            filtered_comment = {
                                'id': comment.get('id'),
                                'author': comment.get('author'),
                                'created_utc': created_utc,
                                'body': comment.get('body'),
                                'score': comment.get('score'),
                                'subreddit': comment.get('subreddit'),
                                'controversiality': comment.get('controversiality')
                            }

                            if last_timestamp_sent is None:
                                current_time_offset = time.time() - created_utc
                            else:
                                time_diff = (created_utc - last_timestamp_sent) / BATCH_SEND_SPEED_FACTOR
                                if time_diff > 0:
                                    time.sleep(time_diff)

                            producer.send(KAFKA_TOPIC, value=filtered_comment)
                            last_timestamp_sent = created_utc

                    except json.JSONDecodeError:
                        print(f"Skipping malformed JSON line: {line[:100]}...")
                    except Exception as e:
                        print(f"Error processing line: {e} - {line[:100]}...")

    except FileNotFoundError:
        print(f"Error: Dataset file not found at {DATA_FILE}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        producer.flush()
        producer.close()
        print("Data provider finished.")

if __name__ == "__main__":
    # Create the topics
    create_kafka_topic(KAFKA_TOPIC, 'kafka:29092', partitions=1, replication_factor=1) 
    create_kafka_topic(KAFKA_TOPIC_PROCESSED, 'kafka:29092', partitions=1, replication_factor=1)

    # Now proceed with sending comments
    process_and_send_comments()