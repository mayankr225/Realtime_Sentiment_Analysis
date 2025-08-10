from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream.connectors import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.common.typeinfo import Types
import json
import time
import re

STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an',
    'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
    'being', 'below', 'between', 'both', 'but', 'by', 'could', 'did', 'do',
    'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further',
    'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself',
    'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', "it's",
    'its', 'itself', 'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor',
    'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our',
    'ours', 'ourselves', 'out', 'over', 'own', 'same', 'she', 'should', 'so',
    'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them',
    'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through',
    'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what',
    'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'with', 'would',
    'you', 'your', 'yours', 'yourself', 'yourselves'
}

# Clean text but keep emojis
def clean_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    cleaned_words = [word for word in words if word.lower() not in STOPWORDS]
    return ' '.join(cleaned_words)

def main():
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)

    kafka_consumer = FlinkKafkaConsumer(
        topics='reddit_comments',
        deserialization_schema=SimpleStringSchema(),
        properties={
            'bootstrap.servers': 'kafka:29092',
            'group.id': 'reddit_flink_processing_group',
            'auto.offset.reset': 'earliest'
        }
    )

    kafka_producer = FlinkKafkaProducer(
        topic='reddit_comments_processed',
        serialization_schema=SimpleStringSchema(),
        producer_config={
            'bootstrap.servers': 'kafka:29092',
            'acks': 'all'
        }
    )

    input_stream = env.add_source(kafka_consumer)

    processed_stream = input_stream \
        .map(lambda json_str: json.loads(json_str)) \
        .filter(lambda data: (
            isinstance(data, dict) and
            'body' in data and
            data['body'] is not None and
            isinstance(data['body'], str) and
            len(data['body'].strip()) >= 10
        )) \
        .map(lambda data: {
            'id': data.get('id'),
            'author': data.get('author'),
            'subreddit': data.get('subreddit'),
            'original_body': data['body'],
            'cleaned_body': clean_text(data['body']),
            'original_score': data.get('score'),
            'created_utc': data.get('created_utc', 0),
            'controversiality': data.get('controversiality', 0)
        }) \
        .map(lambda processed_data: json.dumps(processed_data), output_type=Types.STRING())

    processed_stream.add_sink(kafka_producer)

    print("Starting Flink job...")
    env.execute("Reddit Comments Preprocessing Job (Stopword Removal & Emoji Retention)")
    print("Flink job finished.")

if __name__ == '__main__':
    main()
