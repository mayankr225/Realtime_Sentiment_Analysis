import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import streamlit as st
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import os
import json
from collections import defaultdict

from kafka import KafkaConsumer
from kafka.errors import KafkaError

# --- Constants ---
INIT_ROWS = 100_000
BATCH_SIZE = 50_000

# --- Kafka Configuration ---
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
KAFKA_TOPIC = 'reddit_comments_processed'
KAFKA_CONSUMER_GROUP_ID = 'streamlit_sentiment_group_v1'

# --- Define keys expected in Kafka JSON messages (from Flink's output) ---
ALL_KAFKA_MESSAGE_KEYS = ['cleaned_body', 'original_score', 'subreddit', 'created_utc']

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Live Sentiment Tracker (Dual Keyword)",
    page_icon="📊",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .positive {
        color: #2ecc71; /* Green */
    }
    .neutral {
        color: #f39c12; /* Orange */
    }
    .negative {
        color: #e74c3c; /* Red */
    }
    .keyword-sentiment-box {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        display: inline-block;
        min-width: 48%; /* Adjust for spacing */
        box-sizing: border-box;
    }
</style>
""", unsafe_allow_html=True)

# --- Model Components Initialization (Cached) ---
@st.cache_resource(show_spinner=False)
def initialize_vectorizer_and_label_encoder_and_classifier():
    vectorizer = CountVectorizer(min_df=2, max_features=10000)
    le = LabelEncoder()
    all_classes = ["negative", "neutral", "positive"]
    le.fit(all_classes)
    classifier = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
    return vectorizer, le, classifier

# --- Kafka Consumer Initialization (Cached) ---
@st.cache_resource(show_spinner=False)
def get_kafka_consumer():
    st.info("Connecting to Kafka...")
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=KAFKA_CONSUMER_GROUP_ID,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            auto_commit_interval_ms=5000,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        st.success(f"Successfully subscribed to Kafka topic: `{KAFKA_TOPIC}`")
        return consumer
    except KafkaError as e:
        st.error(f"Failed to connect to Kafka: {e}. Please check KAFKA_BOOTSTRAP_SERVERS and ensure broker is running.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during Kafka consumer initialization: {e}")
        st.stop()

# --- VADER-based Text Labeling ---
def vader_label_texts(texts):
    analyzer = SentimentIntensityAnalyzer()
    data = []
    skipped_count = 0
    for t in texts:
        original_text = str(t) if pd.notna(t) else ""
        processed_text = original_text.strip()

        if not processed_text:
            skipped_count += 1
            continue

        vs = analyzer.polarity_scores(processed_text)
        compound_score = vs['compound']

        if compound_score >= 0.05:
            label = "positive"
        elif compound_score <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        data.append((processed_text, label, vs['neg'], vs['neu'], vs['pos'], vs['compound']))
    return pd.DataFrame(data, columns=["text", "label", "vader_neg", "vader_neu", "vader_pos", "vader_compound"]), skipped_count

# --- Helper to fetch messages from Kafka ---
def fetch_kafka_messages(consumer: KafkaConsumer, num_messages: int, progress_container_passed, timeout_ms: int = 3000) -> list:
    messages_list = []
    polled_count = 0
    # Use the passed container for the progress bar
    progress_bar = progress_container_passed.progress(0, text=f"Fetching up to {num_messages} messages from Kafka...")

    try:
        message_batch_dict = consumer.poll(timeout_ms=timeout_ms, max_records=num_messages)

        for tp, records in message_batch_dict.items():
            for msg in records:
                data = msg.value
                if all(key in data for key in ALL_KAFKA_MESSAGE_KEYS):
                    messages_list.append(data)
                    polled_count += 1
                    progress_bar.progress(
                        min(polled_count / num_messages, 1.0),
                        text=f"Fetching {polled_count}/{num_messages} messages..."
                    )
    except KafkaError as e:
        st.error(f"Kafka error during message fetch: {e}")
    except Exception as e:
        st.warning(f"Error processing Kafka message batch: {e}")
    finally:
        progress_bar.empty() # Clear the progress bar when done or on error
    return messages_list

# --- Main Application Logic ---
def main():
    st.title("📊 Live Sentiment Tracker (Dual Keyword Comparison)")
    st.markdown("Real-time sentiment analysis on data streaming from Kafka, tracking sentiment around **two different keywords** and comparing their trends.")

    # --- UI Placeholders for dynamic updates ---
    # Moved these to the top of the main content area for consistent positioning
    main_status_placeholder = st.empty() # For messages like "Initializing Model..." or "Processing Live Stream..."
    main_info_placeholder = st.empty() # For initial instructions or "No new messages"
    main_progress_bar_placeholder = st.empty()

    chart_placeholder = st.empty()
    current_sentiments_display = st.empty()
    sentiment_pie_charts_container = st.empty()
    subreddit_charts_container = st.empty()

    # Placeholders for the horizontal rule separators
    separator_sentiment_dist = st.empty()
    separator_subreddit_charts = st.empty()


    with st.sidebar:
        st.header("⚙️ Controls")
        keyword1 = st.text_input("Enter Keyword 1:", value="", placeholder="e.g., Bitcoin")
        keyword2 = st.text_input("Enter Keyword 2:", value="", placeholder="e.g., climate")

        col_buttons = st.columns(2)
        with col_buttons[0]:
            run_button = st.button("Start Analysis", type="primary")
        with col_buttons[1]:
            reset_button = st.button("Reset", type="secondary")

        st.markdown("---")
        st.markdown("**Last Record Timestamp:**")
        timestamp_display = st.empty()
        st.markdown("**Processed Comments:**")
        processed_display = st.empty()

    # --- Session State Initialization ---
    if 'vectorizer' not in st.session_state or 'le' not in st.session_state or 'classifier' not in st.session_state:
        st.session_state.vectorizer, st.session_state.le, st.session_state.classifier = initialize_vectorizer_and_label_encoder_and_classifier()
        st.session_state.is_first_run_complete = False
        st.session_state.vader_analyzer = SentimentIntensityAnalyzer()

        st.session_state.keyword_data = {}
        st.session_state.keyword_sentiment_counts = defaultdict(lambda: {'positive': 0, 'neutral': 0, 'negative': 0})
        st.session_state.keyword_subreddit_counts = defaultdict(lambda: defaultdict(int))

        st.session_state.current_processed_count = 0
        st.session_state.last_run_keywords = ["", ""]
        st.session_state.all_possible_classes_array = np.array(st.session_state.le.transform(["negative", "neutral", "positive"]))

        st.session_state.kafka_consumer = get_kafka_consumer()

    keywords_to_track = [keyword1, keyword2]

    # Reset button logic
    if reset_button:
        if 'kafka_consumer' in st.session_state and st.session_state.kafka_consumer is not None:
            try:
                st.session_state.kafka_consumer.close()
                st.info("Kafka consumer closed.")
            except Exception as e:
                st.warning(f"Error closing Kafka consumer: {e}")
        st.session_state.clear()
        st.rerun()

    # --- Check if keywords have changed or if it's a fresh run ---
    if run_button or st.session_state.last_run_keywords != keywords_to_track:
        if st.session_state.last_run_keywords != keywords_to_track:
            st.session_state.keyword_data = {}
            st.session_state.keyword_sentiment_counts = defaultdict(lambda: {'positive': 0, 'neutral': 0, 'negative': 0})
            st.session_state.keyword_subreddit_counts = defaultdict(lambda: defaultdict(int))
            st.session_state.is_first_run_complete = False
            st.session_state.current_processed_count = 0
            st.session_state.vectorizer, st.session_state.le, st.session_state.classifier = initialize_vectorizer_and_label_encoder_and_classifier()
            main_info_placeholder.warning("Analysing new comments for the sentiment......")

        st.session_state.last_run_keywords = keywords_to_track[:]

        for kw in keywords_to_track:
            if kw not in st.session_state.keyword_data:
                st.session_state.keyword_data[kw] = {'timestamps': [], 'confidences': [], 'labels': []}

    # Initial state or if not running
    if not run_button and not st.session_state.is_first_run_complete:
        with main_info_placeholder.container():
            st.info("💡 Enter two keywords and click 'Start Analysis' to begin tracking and comparing sentiment from the Kafka stream!")
            st.write("The application will train a sentiment model on an initial batch of data from Kafka, then analyze incoming comments containing each keyword to show their individual sentiment trends.")
        return
    # Clear the initial info placeholder once the app starts processing
    main_info_placeholder.empty()

    # --- Analysis Logic ---
    if run_button or st.session_state.is_first_run_complete:

        if not st.session_state.is_first_run_complete:
            main_status_placeholder.markdown("### Initializing Model from Kafka Stream...")
            with st.spinner(f'Fetching initial {INIT_ROWS} comments from Kafka for model training...'):
                try:
                    initial_messages = fetch_kafka_messages(st.session_state.kafka_consumer, INIT_ROWS, main_progress_bar_placeholder, timeout_ms=10000)
                    if not initial_messages:
                        main_info_placeholder.error("❌ No messages received from Kafka for initial model training. Please ensure Kafka producer is sending data to the topic and KAFKA_BOOTSTRAP_SERVERS is correct.")
                        st.stop()

                    df_init = pd.DataFrame(initial_messages)
                except Exception as e:
                    st.error(f"Error processing initial Kafka data into DataFrame: {e}")
                    st.stop()

                if not df_init.empty:
                    df_init['created_utc'] = pd.to_datetime(df_init['created_utc'], unit='s')
                    df_init = df_init.sort_values(by='created_utc').reset_index(drop=True)
                    current_batch_timestamp = df_init['created_utc'].iloc[-1]
                else:
                    current_batch_timestamp = pd.to_datetime(time.time(), unit='s')
                    main_info_placeholder.warning("Initial DataFrame from Kafka is empty, using current time as timestamp.")

                texts_init_raw = df_init["cleaned_body"].tolist()
                df_labeled_init, skipped_init = vader_label_texts(texts_init_raw)

                if df_labeled_init.empty:
                    main_info_placeholder.error("❌ No sufficient labeled data found in the initial batch for model training. Please check your data or VADER labeling logic.")
                    st.stop()

                X_init = st.session_state.vectorizer.fit_transform(df_labeled_init["text"].tolist())
                y_init = st.session_state.le.transform(df_labeled_init["label"].tolist())

                if len(np.unique(y_init)) > 1:
                    classes_unique_init = np.unique(y_init)
                    weights_init = compute_class_weight('balanced', classes=classes_unique_init, y=y_init)
                    class_weights_dict_init = dict(zip(classes_unique_init, weights_init))
                    sample_weights_for_fit_init = np.array([class_weights_dict_init[c] for c in y_init])
                else:
                    sample_weights_for_fit_init = None

                if X_init.shape[0] > 0 and X_init.shape[1] > 0:
                    st.session_state.classifier.partial_fit(
                        X_init, y_init,
                        classes=st.session_state.all_possible_classes_array,
                        sample_weight=sample_weights_for_fit_init
                    )

                st.session_state.is_first_run_complete = True
                st.session_state.current_processed_count = len(initial_messages)

                current_latest_sentiments = {}
                for kw in keywords_to_track:
                    keyword_comments_init_df = df_init[
                        df_init['cleaned_body'].str.contains(kw, case=False, na=False)
                    ].copy()

                    keyword_comments_init_df = keyword_comments_init_df.merge(
                        df_labeled_init, left_on='cleaned_body', right_on='text', how='inner'
                    )

                    # NEW: Update sentiment counts and subreddit counts for this keyword
                    if not keyword_comments_init_df.empty:
                        for idx, row in keyword_comments_init_df.iterrows():
                            predicted_label_idx = st.session_state.classifier.predict(st.session_state.vectorizer.transform([row['text']]))[0]
                            predicted_label = st.session_state.le.inverse_transform([predicted_label_idx])[0]
                            st.session_state.keyword_sentiment_counts[kw][predicted_label] += 1

                            subreddit = row['subreddit']
                            if subreddit:
                                st.session_state.keyword_subreddit_counts[kw][subreddit] += 1

                    leading_confidence = 0.5
                    leading_label = "neutral"

                    if not keyword_comments_init_df.empty and hasattr(st.session_state.classifier, 'coef_'):
                        X_keyword_comments_init = st.session_state.vectorizer.transform(keyword_comments_init_df['text'].tolist())

                        if hasattr(st.session_state.classifier, 'classes_') and len(st.session_state.classifier.classes_) > 1:
                            keyword_comment_probs_init = st.session_state.classifier.predict_proba(X_keyword_comments_init)
                            avg_probs_init = np.mean(keyword_comment_probs_init, axis=0)

                            leading_idx = np.argmax(avg_probs_init)
                            leading_confidence = avg_probs_init[leading_idx]
                            leading_label = st.session_state.le.inverse_transform([leading_idx])[0]

                    st.session_state.keyword_data[kw]['confidences'].append(leading_confidence)
                    st.session_state.keyword_data[kw]['timestamps'].append(current_batch_timestamp)
                    st.session_state.keyword_data[kw]['labels'].append(leading_label)

                    current_latest_sentiments[kw] = {'confidence': leading_confidence, 'label': leading_label}

                update_display(
                    st.session_state.keyword_data,
                    chart_placeholder,
                    timestamp_display,
                    processed_display,
                    current_sentiments_display,
                    sentiment_pie_charts_container,
                    subreddit_charts_container,
                    separator_sentiment_dist,
                    separator_subreddit_charts,
                    current_batch_timestamp,
                    st.session_state.current_processed_count,
                    current_latest_sentiments,
                    st.session_state.keyword_sentiment_counts,
                    st.session_state.keyword_subreddit_counts
                )
            main_status_placeholder.markdown("### Processing Live Stream...")
            main_info_placeholder.empty() # Clear the info placeholder once live processing starts

        # --- Main Loop for Live Stream Processing (Runs indefinitely) ---
        while True:
            try:
                batch_messages = fetch_kafka_messages(st.session_state.kafka_consumer, BATCH_SIZE, main_progress_bar_placeholder)

                current_batch_timestamp = pd.to_datetime(time.time(), unit='s')

                if not batch_messages:
                    main_info_placeholder.info(f"No new Kafka messages. Waiting for data...")
                    time.sleep(2)
                    current_latest_sentiments = {kw: {'confidence': st.session_state.keyword_data[kw]['confidences'][-1] if st.session_state.keyword_data[kw]['confidences'] else 0.5,
                                                       'label': st.session_state.keyword_data[kw]['labels'][-1] if st.session_state.keyword_data[kw]['labels'] else "neutral"}
                                                 for kw in keywords_to_track}

                    update_display(
                        st.session_state.keyword_data,
                        chart_placeholder,
                        timestamp_display,
                        processed_display,
                        current_sentiments_display,
                        sentiment_pie_charts_container,
                        subreddit_charts_container,
                        separator_sentiment_dist,
                        separator_subreddit_charts,
                        current_batch_timestamp,
                        st.session_state.current_processed_count,
                        current_latest_sentiments,
                        st.session_state.keyword_sentiment_counts,
                        st.session_state.keyword_subreddit_counts
                    )
                    continue
                else:
                    main_info_placeholder.empty() # Clear the info message if messages are received

                df_batch = pd.DataFrame(batch_messages)
                st.session_state.current_processed_count += len(batch_messages)

                df_batch['created_utc'] = pd.to_datetime(df_batch['created_utc'], unit='s')
                df_batch = df_batch.sort_values(by='created_utc').reset_index(drop=True)
                current_batch_timestamp = df_batch['created_utc'].iloc[-1]

                texts_raw_batch = df_batch["cleaned_body"].tolist()
                df_labeled_batch, skipped_batch = vader_label_texts(texts_raw_batch)

                if df_labeled_batch.empty:
                    for kw in keywords_to_track:
                        if st.session_state.keyword_data[kw]['confidences']:
                            st.session_state.keyword_data[kw]['confidences'].append(st.session_state.keyword_data[kw]['confidences'][-1])
                            st.session_state.keyword_data[kw]['timestamps'].append(current_batch_timestamp)
                            st.session_state.keyword_data[kw]['labels'].append(st.session_state.keyword_data[kw]['labels'][-1])
                        else:
                            st.session_state.keyword_data[kw]['confidences'].append(0.5)
                            st.session_state.keyword_data[kw]['timestamps'].append(current_batch_timestamp)
                            st.session_state.keyword_data[kw]['labels'].append("neutral")

                    current_latest_sentiments = {kw: {'confidence': st.session_state.keyword_data[kw]['confidences'][-1],
                                                       'label': st.session_state.keyword_data[kw]['labels'][-1]}
                                                 for kw in keywords_to_track}

                    update_display(
                        st.session_state.keyword_data,
                        chart_placeholder,
                        timestamp_display,
                        processed_display,
                        current_sentiments_display,
                        sentiment_pie_charts_container,
                        subreddit_charts_container,
                        separator_sentiment_dist,
                        separator_subreddit_charts,
                        current_batch_timestamp,
                        st.session_state.current_processed_count,
                        current_latest_sentiments,
                        st.session_state.keyword_sentiment_counts,
                        st.session_state.keyword_subreddit_counts
                    )
                    time.sleep(0.1)
                    continue

                X_batch_new = st.session_state.vectorizer.transform(df_labeled_batch["text"].tolist())
                y_batch_new = st.session_state.le.transform(df_labeled_batch["label"].tolist())

                if len(np.unique(y_batch_new)) > 1:
                    classes_unique_batch = np.unique(y_batch_new)
                    weights_batch = compute_class_weight('balanced', classes=classes_unique_batch, y=y_batch_new)
                    class_weights_dict_batch = dict(zip(classes_unique_batch, weights_batch))
                    sample_weights_for_fit_batch = np.array([class_weights_dict_batch[c] for c in y_batch_new])
                else:
                    sample_weights_for_fit_batch = None

                if X_batch_new.shape[0] > 0 and X_batch_new.shape[1] > 0 and hasattr(st.session_state.classifier, 'classes_'):
                    st.session_state.classifier.partial_fit(
                        X_batch_new, y_batch_new,
                        classes=st.session_state.all_possible_classes_array,
                        sample_weight=sample_weights_for_fit_batch
                    )
                elif X_batch_new.shape[0] > 0 and X_batch_new.shape[1] > 0 and not hasattr(st.session_state.classifier, 'classes_'):
                    st.session_state.classifier.partial_fit(
                        X_batch_new, y_batch_new,
                        classes=st.session_state.all_possible_classes_array,
                        sample_weight=sample_weights_for_fit_batch
                    )

                current_latest_sentiments = {}
                for kw in keywords_to_track:
                    keyword_comments_batch_df = df_batch[
                        df_batch['cleaned_body'].str.contains(kw, case=False, na=False)
                    ].copy()

                    keyword_comments_batch_df = keyword_comments_batch_df.merge(
                        df_labeled_batch, left_on='cleaned_body', right_on='text', how='inner'
                    )

                    # NEW: Update sentiment counts and subreddit counts for this keyword
                    if not keyword_comments_batch_df.empty:
                        for idx, row in keyword_comments_batch_df.iterrows():
                            predicted_label_idx = st.session_state.classifier.predict(st.session_state.vectorizer.transform([row['text']]))[0]
                            predicted_label = st.session_state.le.inverse_transform([predicted_label_idx])[0]
                            st.session_state.keyword_sentiment_counts[kw][predicted_label] += 1

                            subreddit = row['subreddit']
                            if subreddit:
                                st.session_state.keyword_subreddit_counts[kw][subreddit] += 1


                    leading_confidence = None
                    leading_label = None

                    if not keyword_comments_batch_df.empty and hasattr(st.session_state.classifier, 'coef_') and len(st.session_state.classifier.classes_) > 1:
                        X_keyword_comments = st.session_state.vectorizer.transform(keyword_comments_batch_df['text'].tolist())

                        keyword_comment_probs = st.session_state.classifier.predict_proba(X_keyword_comments)
                        avg_probs = np.mean(keyword_comment_probs, axis=0)

                        leading_idx = np.argmax(avg_probs)
                        leading_confidence = avg_probs[leading_idx]
                        leading_label = st.session_state.le.inverse_transform([leading_idx])[0]
                    else:
                        if st.session_state.keyword_data[kw]['confidences']:
                            leading_confidence = st.session_state.keyword_data[kw]['confidences'][-1]
                            leading_label = st.session_state.keyword_data[kw]['labels'][-1]
                        else:
                            leading_confidence = 0.5
                            leading_label = "neutral"

                    st.session_state.keyword_data[kw]['confidences'].append(leading_confidence)
                    st.session_state.keyword_data[kw]['timestamps'].append(current_batch_timestamp)
                    st.session_state.keyword_data[kw]['labels'].append(leading_label)

                    current_latest_sentiments[kw] = {'confidence': leading_confidence, 'label': leading_label}

                update_display(
                    st.session_state.keyword_data,
                    chart_placeholder,
                    timestamp_display,
                    processed_display,
                    current_sentiments_display,
                    sentiment_pie_charts_container,
                    subreddit_charts_container,
                    separator_sentiment_dist,
                    separator_subreddit_charts,
                    current_batch_timestamp,
                    st.session_state.current_processed_count,
                    current_latest_sentiments,
                    st.session_state.keyword_sentiment_counts,
                    st.session_state.keyword_subreddit_counts
                )

                time.sleep(0.1)

            except KafkaError as e:
                main_info_placeholder.error(f"Kafka error encountered: {e}. Please ensure broker is running and topic `{KAFKA_TOPIC}` exists. Retrying on next cycle...")
                time.sleep(5)
                continue
            except Exception as e:
                main_info_placeholder.error(f"An unexpected error occurred during stream processing: {str(e)}. Attempting to continue.")
                for kw in keywords_to_track:
                    if st.session_state.keyword_data[kw]['confidences']:
                        st.session_state.keyword_data[kw]['confidences'].append(st.session_state.keyword_data[kw]['confidences'][-1])
                        st.session_state.keyword_data[kw]['timestamps'].append(current_batch_timestamp if 'current_batch_timestamp' in locals() else pd.to_datetime(time.time(), unit='s'))
                        st.session_state.keyword_data[kw]['labels'].append(st.session_state.keyword_data[kw]['labels'][-1])
                    else:
                        st.session_state.keyword_data[kw]['confidences'].append(0.5)
                        st.session_state.keyword_data[kw]['timestamps'].append(current_batch_timestamp if 'current_batch_timestamp' in locals() else pd.to_datetime(time.time(), unit='s'))
                        st.session_state.keyword_data[kw]['labels'].append("neutral")
                    current_latest_sentiments[kw] = {'confidence': st.session_state.keyword_data[kw]['confidences'][-1],
                                                      'label': st.session_state.keyword_data[kw]['labels'][-1]}

                update_display(
                    st.session_state.keyword_data,
                    chart_placeholder,
                    timestamp_display,
                    processed_display,
                    current_sentiments_display,
                    sentiment_pie_charts_container,
                    subreddit_charts_container,
                    separator_sentiment_dist,
                    separator_subreddit_charts,
                    current_batch_timestamp if 'current_batch_timestamp' in locals() else pd.to_datetime(time.time(), unit='s'),
                    st.session_state.current_processed_count,
                    current_latest_sentiments,
                    st.session_state.keyword_sentiment_counts,
                    st.session_state.keyword_subreddit_counts
                )
                time.sleep(0.1)
                continue

# --- Display Update Function ---
def update_display(
    keyword_data,
    chart_placeholder,
    timestamp_display,
    processed_display,
    current_sentiments_display_placeholder,
    sentiment_pie_charts_container_passed,
    subreddit_charts_container_passed,
    separator_sentiment_dist_passed,
    separator_subreddit_charts_passed,
    current_timestamp,
    processed_count,
    current_latest_sentiments,
    keyword_sentiment_counts,
    keyword_subreddit_counts
):
    timestamp_display.markdown(f"`{current_timestamp.strftime('%Y-%m-%d %H:%M:%S')}`")
    processed_display.markdown(f"`{processed_count:,}`")

    all_raw_chart_data = []
    for kw, data in keyword_data.items():
        if not data['timestamps']:
            continue
        df_kw = pd.DataFrame({
            'Timestamp': pd.to_datetime(data['timestamps']),
            'Confidence': data['confidences'],
            'Leading Sentiment': data['labels']
        })
        df_kw['Keyword'] = kw
        all_raw_chart_data.append(df_kw)

    chart_data_for_plot = pd.DataFrame()
    if all_raw_chart_data:
        raw_combined_df = pd.concat(all_raw_chart_data, ignore_index=True)

        resampled_dfs = []
        for kw in raw_combined_df['Keyword'].unique():
            df_for_resample = raw_combined_df[raw_combined_df['Keyword'] == kw].set_index('Timestamp')

            df_resampled_kw = df_for_resample.resample('5min').agg({
                'Confidence': 'mean',
                'Leading Sentiment': lambda x: x.mode()[0] if not x.empty else None
            }).reset_index()
            df_resampled_kw['Keyword'] = kw
            resampled_dfs.append(df_resampled_kw)

        if resampled_dfs:
            chart_data_for_plot = pd.concat(resampled_dfs, ignore_index=True)
            chart_data_for_plot.dropna(subset=['Confidence'], inplace=True)
        else:
            chart_data_for_plot = pd.DataFrame(columns=['Timestamp', 'Confidence', 'Leading Sentiment', 'Keyword'])

    keyword_names = list(keyword_data.keys())
    keyword_colors = {}
    if len(keyword_names) >= 1:
        keyword_colors[keyword_names[0]] = '#1f77b4'
    if len(keyword_names) >= 2:
        keyword_colors[keyword_names[1]] = '#ff7f0e'

    fig = px.line(
        chart_data_for_plot,
        x='Timestamp',
        y='Confidence',
        color='Keyword',
        title="Leading Sentiment Confidence Trend for Keywords' Context (5-min Average)",
        labels={'Confidence': 'Average Confidence Score', 'Timestamp': 'Time of Last Record'},
        height=400,
        hover_data={'Leading Sentiment': True, 'Confidence': ':.1%', 'Keyword': False, 'Timestamp': '|%Y-%m-%d %H:%M:%S'},
        color_discrete_map=keyword_colors if len(keyword_names) == 2 else None
    )

    fig.update_layout(
        yaxis_tickformat=".0%",
        hovermode="x unified",
        legend_title_text="Keyword",
        xaxis_title="Time of Last Record",
        yaxis_title="Average Confidence Score",
        yaxis=dict(range=[0, 1])
    )

    # Set up sentiment color map for markers
    sentiment_color_map = {
        'positive': '#2ecc71',
        'neutral': '#f39c12',
        'negative': '#e74c3c'
    }

    # Iterate through each trace (keyword line) to set marker properties individually
    if not chart_data_for_plot.empty:
        for trace_idx, trace in enumerate(fig.data):
            # Ensure the mode includes markers
            fig.data[trace_idx].mode = "lines+markers"
            
            # Get the keyword for this trace
            current_keyword = trace.name # trace.name holds the value from the 'color' mapping, which is 'Keyword'

            # Filter chart_data_for_plot for this specific keyword to get its sentiment data
            df_for_trace_sentiment = chart_data_for_plot[chart_data_for_plot['Keyword'] == current_keyword]

            if not df_for_trace_sentiment.empty:
                # Create a list of colors based on the 'Leading Sentiment' for each data point of this keyword
                marker_colors = [sentiment_color_map.get(s, '#808080') for s in df_for_trace_sentiment['Leading Sentiment']]

                # Update the marker properties for this specific trace with sentiment-based colors
                fig.data[trace_idx].update(
                    marker=dict(
                        size=8,
                        color=marker_colors # Assign the list of colors
                    )
                )

    chart_placeholder.plotly_chart(
        fig,
        use_container_width=True,
        key=f"sentiment_chart_{processed_count}"
    )

    sentiment_html = "<h4>Current Keyword Sentiments:</h4>"
    cols_sentiment = current_sentiments_display_placeholder.columns(len(current_latest_sentiments))

    for idx, (kw, sentiment_info) in enumerate(current_latest_sentiments.items()):
        cols_sentiment[idx].markdown(
            f"<div class='keyword-sentiment-box'><b>{kw}:</b> "
            f"<span class='{sentiment_info['label']}'>{sentiment_info['label'].capitalize()}</span> "
            f"(Confidence: {sentiment_info['confidence']:.1%})</div>",
            unsafe_allow_html=True
        )

    separator_sentiment_dist_passed.markdown("---")

    # --- Sentiment Distribution Pie Charts ---
    with sentiment_pie_charts_container_passed.container():
        sentiment_pie_cols = st.columns(len(keyword_data.keys()))
        for idx, kw in enumerate(keyword_data.keys()):
            with sentiment_pie_cols[idx]:
                st.markdown(f"<h5>Sentiment Distribution for '{kw}'</h5>", unsafe_allow_html=True)
                sentiment_df = pd.DataFrame([
                    {'Sentiment': 'Positive', 'Count': keyword_sentiment_counts[kw]['positive']},
                    {'Sentiment': 'Neutral', 'Count': keyword_sentiment_counts[kw]['neutral']},
                    {'Sentiment': 'Negative', 'Count': keyword_sentiment_counts[kw]['negative']}
                ])
                sentiment_df = sentiment_df[sentiment_df['Count'] > 0]

                if not sentiment_df.empty:
                    fig_pie = px.pie(
                        sentiment_df,
                        values='Count',
                        names='Sentiment',
                        title=f"'{kw}'",
                        color='Sentiment',
                        color_discrete_map={
                            'Positive': '#2ecc71',
                            'Neutral': '#f39c12',
                            'Negative': '#e74c3c'
                        },
                        hole=0.4
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(showlegend=False)
                    sentiment_pie_cols[idx].plotly_chart(fig_pie, use_container_width=True, key=f"pie_chart_{kw}_{processed_count}")
                else:
                    sentiment_pie_cols[idx].info("No sentiment data yet for this keyword.")

    separator_subreddit_charts_passed.markdown("---")

    # --- Top 5 Subreddits Bar Charts ---
    with subreddit_charts_container_passed.container():
        subreddit_chart_cols = st.columns(len(keyword_data.keys()))
        for idx, kw in enumerate(keyword_data.keys()):
            with subreddit_chart_cols[idx]:
                st.markdown(f"<h5>Top 5 Subreddits for '{kw}'</h5>", unsafe_allow_html=True)
                subreddit_counts = keyword_subreddit_counts[kw]
                if subreddit_counts:
                    top_subreddits = sorted(subreddit_counts.items(), key=lambda item: item[1], reverse=True)[:5]
                    df_subreddits = pd.DataFrame(top_subreddits, columns=['Subreddit', 'Mentions'])

                    fig_bar = px.bar(
                        df_subreddits,
                        x='Mentions',
                        y='Subreddit',
                        orientation='h',
                        title=f"'{kw}'",
                        color='Mentions',
                        color_continuous_scale=px.colors.sequential.Viridis
                    )
                    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                    subreddit_chart_cols[idx].plotly_chart(fig_bar, use_container_width=True, key=f"bar_chart_{kw}_{processed_count}")
                else:
                    subreddit_chart_cols[idx].info("No subreddit data yet for this keyword.")


if __name__ == "__main__":
    main()