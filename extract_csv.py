from kafka import KafkaConsumer
import json
import csv

consumer = KafkaConsumer(
    'reddit_comments_processed',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)
print("Starting Extraction from Kafka...")
max_comments = 100_000
count = 0
with open('cleaned_comments.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['subreddit', 'cleaned_text','score','created_utc','controversiality'])
    writer.writeheader()
    for message in consumer:
        if count >= max_comments:
            break
        data = message.value
        writer.writerow({
            'subreddit': data.get('subreddit'),
            'cleaned_text': data.get('cleaned_body',''),
            'score': data.get('original_score'),
            'created_utc': data.get('created_utc', 0),
            'controversiality': data.get('controversiality', 0)
        })
        count += 1
print("Extraction completed. Data written to cleaned_comments.csv")