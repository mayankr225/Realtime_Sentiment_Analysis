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
import plotly.graph_objects as go
import json
from matplotlib.colors import ListedColormap 
from kafka import KafkaConsumer, TopicPartition
from kafka.errors import KafkaError

# --- Constants ---
INIT_ROWS = 100_000 
BATCH_SIZE = 5_000 

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
    """Initializes CountVectorizer, LabelEncoder, and SGDClassifier once."""
    vectorizer = CountVectorizer(min_df=2, max_features=10000) 
    le = LabelEncoder()
    all_classes = ["negative", "neutral", "positive"]
    le.fit(all_classes) 
    classifier = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
    return vectorizer, le, classifier

# --- Kafka Consumer Initialization (No @st.cache_resource for direct control) ---
def get_kafka_consumer(): 
    """Initializes and returns a Kafka consumer using kafka-python. Returns None on failure."""
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
        return None # Return None to indicate failure, let calling code handle st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during Kafka consumer initialization: {e}")
        return None # Return None to indicate failure

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
def fetch_kafka_messages(consumer: KafkaConsumer, num_messages: int, timeout_ms: int = 3000) -> list:
    messages_list = []
    polled_count = 0
    progress_placeholder = st.empty() 
    progress_bar = progress_placeholder.progress(0, text=f"Fetching up to {num_messages} messages from Kafka...")

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

    progress_bar.progress(1.0, text=f"Finished fetching {polled_count} messages from Kafka.")
    time.sleep(0.5) 
    progress_placeholder.empty() 
    return messages_list

# --- Sentiment Distribution Chart Function ---
def display_sentiment_distribution(all_comments_df, keywords_to_track, placeholder_for_pie_charts): # Added placeholder argument
    if all_comments_df.empty:
        placeholder_for_pie_charts.empty() # Clear the placeholder if no data
        st.info("DEBUG (Sentiment Dist): Input all_comments_df is empty, nothing to plot.")
        return

    all_charts = []
    
    for kw in keywords_to_track:
        if not kw: 
            continue

        keyword_comments_df = all_comments_df[
            all_comments_df['text'].str.contains(kw, case=False, na=False)
        ]
        
        if not keyword_comments_df.empty:
            sentiment_counts = keyword_comments_df['sgd_label'].value_counts(normalize=True).reset_index()
            sentiment_counts.columns = ['Sentiment', 'Percentage']
            sentiment_counts['Percentage'] = sentiment_counts['Percentage'] * 100 

            full_labels = pd.DataFrame({'Sentiment': ['positive', 'neutral', 'negative'], 'Percentage': 0.0})
            
            sentiment_counts = pd.concat([full_labels, sentiment_counts]).drop_duplicates(subset=['Sentiment'], keep='last').reset_index(drop=True)
            
            sentiment_counts['Percentage'] = sentiment_counts.apply(
                lambda row: row['Percentage'] if row['Percentage'] > 0 else 0.0, axis=1
            )
            
            sentiment_counts['Sentiment'] = pd.Categorical(sentiment_counts['Sentiment'], categories=['positive', 'neutral', 'negative'], ordered=True)
            sentiment_counts = sentiment_counts.sort_values('Sentiment')

            if sentiment_counts['Percentage'].sum() == 0 and len(sentiment_counts) > 0:
                st.warning(f"DEBUG (Sentiment Dist): All sentiment percentages for '{kw}' are zero. Chart might appear blank.")
            
            sentiment_colors = {
                "positive": "#2ecc71",
                "neutral": "#f39c12",
                "negative": "#e74c3c"
            }

            fig_pie = px.pie(
                sentiment_counts,
                values='Percentage',
                names='Sentiment',
                title=f'Cumulative Sentiment Distribution for "{kw}"', 
                color='Sentiment',
                color_discrete_map=sentiment_colors,
                hole=0.3 
            )
            fig_pie.update_traces(textinfo='percent+label')
            fig_pie.update_layout(
                margin=dict(l=20, r=20, t=60, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            all_charts.append(fig_pie)

    if all_charts:
        with placeholder_for_pie_charts.container(): 
            st.markdown("---")
            st.markdown("### Cumulative Sentiment Breakdown") 
            cols_pie_charts = st.columns(len(all_charts)) 
            for i, chart in enumerate(all_charts):
                with cols_pie_charts[i]:
                    st.plotly_chart(chart, use_container_width=True, key=f"pie_chart_{keywords_to_track[i]}_{i}_{st.session_state.loop_counter}")
    else:
        placeholder_for_pie_charts.empty() # Clear if no charts to display


# --- Display Update Function ---
def update_display(
    keyword_data,                 
    chart_placeholder,
    comment_volume_placeholder,
    timestamp_display,            
    processed_display,
    current_sentiments_display_placeholder,
    current_timestamp,            
    processed_count,
    current_latest_sentiments,
    keywords_to_track 
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
        raw_combined_df['Timestamp'] = pd.to_datetime(raw_combined_df['Timestamp']) # Ensure datetime type
        chart_data_for_plot = raw_combined_df.sort_values(by='Timestamp').copy()
        
        chart_data_for_plot.dropna(subset=['Confidence'], inplace=True)
    
    sentiment_marker_colors = {
        "positive": "lightgreen", 
        "neutral": "gold",  
        "negative": "red"  
    }

    keyword_line_colors = {}
    if len(keywords_to_track) >= 1:
        keyword_line_colors[keywords_to_track[0]] = 'darkblue'  
    if len(keywords_to_track) >= 2:
        keyword_line_colors[keywords_to_track[1]] = 'gray'  

    # Sentiment Trend Chart
    fig = go.Figure()

    if not chart_data_for_plot.empty:
        for kw_name in chart_data_for_plot['Keyword'].unique():
            df_kw_plot = chart_data_for_plot[chart_data_for_plot['Keyword'] == kw_name].copy()
            
            df_kw_plot['MarkerColor'] = df_kw_plot['Leading Sentiment'].map(sentiment_marker_colors)

            df_kw_plot['Sentiment Score (%)'] = df_kw_plot['Confidence'] * 100

            fig.add_trace(go.Scatter(
                x=df_kw_plot["Timestamp"],
                y=df_kw_plot["Sentiment Score (%)"], 
                mode="markers+lines", 
                marker=dict(
                    color=df_kw_plot["MarkerColor"], 
                    size=10, 
                    line=dict(width=1, color='DarkSlateGrey') 
                ),
                line=dict(
                    color=keyword_line_colors.get(kw_name, 'grey'), 
                    width=2
                ),
                name=f"{kw_name} Sentiment",
                hovertemplate=(
                    "<b>Timestamp</b>: %{x|%Y-%m-%d %H:%M}<br>" + 
                    "<b>Sentiment Score</b>: %{y:.1f}%<br>" +
                    "<b>Dominant Sentiment</b>: %{customdata}<extra></extra>"
                ),
                customdata=df_kw_plot["Leading Sentiment"] 
            ))

    fig.update_layout(
        title="Leading Sentiment Confidence Trend for Keyword's context (5-min average)", 
        xaxis_title="Timestamp",
        yaxis_title="Average Confidence Score (%)", 
        yaxis=dict(range=[0, 105], dtick=10), 
        hovermode="x unified",
        legend_title_text="Keyword", 
        margin=dict(l=0, r=0, t=50, b=0), 
    )
    
    chart_placeholder.plotly_chart(
        fig,
        use_container_width=True,
        key=f"sentiment_trend_chart_{st.session_state.loop_counter}"
    )

    # Comment Volume Bar Chart (Cumulative Total per Keyword)
    bar_fig = go.Figure()
    
    if 'cumulative_comments_df' in st.session_state and not st.session_state.cumulative_comments_df.empty:
        cumulative_counts_data = []
        for kw in keywords_to_track:
            if kw:
                # Count comments containing the keyword in the *cumulative* dataframe
                total_comments_for_kw = st.session_state.cumulative_comments_df[
                    st.session_state.cumulative_comments_df['text'].str.contains(kw, case=False, na=False)
                ].shape[0]
                cumulative_counts_data.append({'Keyword': kw, 'Total Comments': total_comments_for_kw})

        if cumulative_counts_data:
            df_cumulative_counts = pd.DataFrame(cumulative_counts_data)

            keyword_bar_colors = {}
            if len(keywords_to_track) >= 1:
                keyword_bar_colors[keywords_to_track[0]] = 'darkblue'
            if len(keywords_to_track) >= 2:
                keyword_bar_colors[keywords_to_track[1]] = 'grey'

            bar_fig.add_trace(go.Bar(
                x=df_cumulative_counts["Keyword"],
                y=df_cumulative_counts["Total Comments"],
                marker_color=[keyword_bar_colors.get(kw, 'lightgrey') for kw in df_cumulative_counts["Keyword"]],
                text=df_cumulative_counts["Total Comments"], 
                textposition='outside', 
                name="Total Comments"
            ))
            
            bar_fig.update_layout(
                title="Cumulative Comment Volume Per Keyword",
                xaxis_title="Keyword",
                yaxis_title="Total Number of Comments",
                margin=dict(l=0, r=0, t=50, b=0),
                showlegend=False 
            )
        else:
            bar_fig.update_layout(
                title="Cumulative Comment Volume Per Keyword",
                annotations=[
                    dict(text="No comments found for selected keywords.", 
                         xref="paper", yref="paper", showarrow=False, font=dict(size=16))
                ]
            )
    else:
        bar_fig.update_layout(
            title="Cumulative Comment Volume Per Keyword",
            annotations=[
                dict(text="No comments processed yet.", 
                     xref="paper", yref="paper", showarrow=False, font=dict(size=16))
            ]
        )
    
    # Render the bar chart in its designated placeholder
    comment_volume_placeholder.plotly_chart(bar_fig, use_container_width=True, key=f"comment_volume_chart_{st.session_state.loop_counter}")


    sentiment_html = "<h4>Current Keyword Sentiments:</h4>"
    cols_sentiment = current_sentiments_display_placeholder.columns(len(current_latest_sentiments))
    
    for idx, (kw, sentiment_info) in enumerate(current_latest_sentiments.items()):
        cols_sentiment[idx].markdown(
            f"<div class='keyword-sentiment-box'><b>{kw}:</b> "
            f"<span class='{sentiment_info['label']}'>{sentiment_info['label'].capitalize()}</span> "
            f"(Confidence: {sentiment_info['confidence']:.1%})</div>",
            unsafe_allow_html=True
        )

# --- Function to display Top Subreddits ---
def display_top_subreddits(all_comments_df, keywords_to_track, placeholder_for_subreddits):
    if all_comments_df.empty:
        placeholder_for_subreddits.empty()
        return

    with placeholder_for_subreddits.container():
        st.markdown("### Top Subreddits by Keyword")
        
        cols_subreddits = st.columns(len(keywords_to_track))

        for idx, kw in enumerate(keywords_to_track):
            if not kw:
                continue
            
            with cols_subreddits[idx]:
                st.subheader(f"Subreddits for \"{kw}\"")
                keyword_comments_df = all_comments_df[
                    all_comments_df['text'].str.contains(kw, case=False, na=False)
                ]

                if not keyword_comments_df.empty:
                    top_subreddits = keyword_comments_df['subreddit'].value_counts().reset_index()
                    top_subreddits.columns = ['Subreddit', 'Comment Count']
                    
                    # Display top 5 subreddits
                    st.dataframe(top_subreddits.head(5), hide_index=True, use_container_width=True)
                else:
                    st.info(f"No comments found for '{kw}' to determine top subreddits.")

# --- Main Application Logic ---
def main():
    st.title("Live Sentiment Tracker (Dual Keyword Comparison)")
    st.markdown("Real-time sentiment analysis on data streaming from Kafka, tracking sentiment around **two different keywords** and comparing their trends.")
    
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
    
    # Initialize placeholders at the top level of the main function
    st.markdown("### Sentiment Trends Over Time")
    chart_placeholder = st.empty() # This is for the sentiment trend line chart
    current_sentiments_display = st.empty()
    st.markdown("---")

    # Create columns for layout
    col_left, col_right = st.columns([1, 1]) 

    # Place the bar chart in the left column
    with col_left:
        st.markdown("### Comment Volume Over Time")
        comment_volume_placeholder = st.empty() # For the cumulative bar chart

    # Place the Top Subreddits in the right column
    with col_right:
        subreddit_placeholder = st.empty() # For top subreddits

    # Place the cumulative pie charts below the two columns
    sentiment_distribution_placeholder = st.empty() 

    # --- Session State Initialization ---
    # `cache_buster` is no longer needed as st.cache_resource is removed for KafkaConsumer
    # if 'cache_buster' not in st.session_state:
    #     st.session_state.cache_buster = 0 

    if 'vectorizer' not in st.session_state or 'le' not in st.session_state or 'classifier' not in st.session_state:
        st.session_state.vectorizer, st.session_state.le, st.session_state.classifier = initialize_vectorizer_and_label_encoder_and_classifier()
        st.session_state.is_first_run_complete = False
        st.session_state.vader_analyzer = SentimentIntensityAnalyzer() 
        
        st.session_state.keyword_data = {} # Initialize as empty dict
        
        st.session_state.current_processed_count = 0
        st.session_state.last_run_keywords = ["", ""] 
        st.session_state.all_possible_classes_array = np.array(st.session_state.le.transform(["negative", "neutral", "positive"]))
        
        # Initialize loop_counter here
        st.session_state.loop_counter = 0

        # Initialize cumulative DataFrame for sentiment distribution
        st.session_state.cumulative_comments_df = pd.DataFrame(columns=['text', 'original_score', 'subreddit', 'created_utc', 'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'sgd_label', 'sgd_confidence'])

        # Initialize for CountVectorizer vocabulary consistency
        st.session_state.vectorizer_vocabulary = None
        st.session_state.expected_features_count = 0

    # --- Reset button logic ---
    if reset_button:
        # Clear the @st.cache_resource for model components
        initialize_vectorizer_and_label_encoder_and_classifier.clear()
        # Modified message to suggest waiting
        st.info("🔄 Resetting application... Please wait a moment for re-initialization.")

        if 'kafka_consumer' in st.session_state and st.session_state.kafka_consumer is not None:
            try:
                st.session_state.kafka_consumer.close()
                st.info("Kafka consumer closed.")
            except Exception as e:
                st.warning(f"Error closing Kafka consumer: {e}")
            st.session_state.kafka_consumer = None # Explicitly set to None

        # Clear specific session state items that hold data (optional, but good for a clean reset)
        for key in ['vectorizer', 'le', 'classifier', 'is_first_run_complete', 'vader_analyzer',
                     'keyword_data', 'current_processed_count', 'last_run_keywords',
                     'all_possible_classes_array', 'loop_counter', 'cumulative_comments_df',
                     'vectorizer_vocabulary', 'expected_features_count']: # Added new keys
            if key in st.session_state:
                del st.session_state[key]
        
        st.rerun() # Rerun the app to re-initialize everything

    # List of keywords currently being tracked
    keywords_to_track = [keyword1, keyword2]

    # --- Initialize keyword_data structure for each tracked keyword ---
    # This ensures that each keyword has its corresponding dictionary with empty lists
    # before we try to append to them, preventing IndexError.
    for kw in keywords_to_track:
        if kw and kw not in st.session_state.keyword_data:
            st.session_state.keyword_data[kw] = {
                'timestamps': [],
                'confidences': [],
                'labels': [],
                'counts': []
            }
    
    # Clean up keyword_data for keywords no longer being tracked (optional, but good practice)
    keys_to_remove = [kw for kw in st.session_state.keyword_data if kw not in keywords_to_track]
    for key in keys_to_remove:
        del st.session_state.keyword_data[key]

    # --- Kafka Consumer Initialization/Re-initialization ---
    # Manage consumer directly in session_state, without @st.cache_resource
    if 'kafka_consumer' not in st.session_state or st.session_state.kafka_consumer is None or st.session_state.kafka_consumer._closed:
        st.session_state.kafka_consumer = get_kafka_consumer()
        if st.session_state.kafka_consumer is None:
            st.error("Failed to initialize/re-initialize Kafka consumer. Please check your Kafka setup.")
            st.stop()


    # --- Analysis Logic ---
    if run_button or st.session_state.is_first_run_complete:
        
        if not st.session_state.is_first_run_complete:
            with st.spinner(f'Fetching initial {INIT_ROWS} comments from Kafka for model training...'):
                try:
                    initial_messages = fetch_kafka_messages(st.session_state.kafka_consumer, INIT_ROWS, timeout_ms=10000) 
                    if not initial_messages:
                        st.error("❌ No messages received from Kafka for initial model training. Please ensure Kafka producer is sending data to the topic and KAFKA_BOOTSTRAP_SERVERS is correct.")
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
                    st.warning("Initial DataFrame from Kafka is empty, using current time as timestamp.")

                texts_init_raw = df_init["cleaned_body"].tolist()
                df_labeled_init, skipped_init = vader_label_texts(texts_init_raw)
                
                if df_labeled_init.empty:
                    st.error("❌ No sufficient labeled data found in the initial batch for model training. Please check your data or VADER labeling logic.")
                    st.stop()
                
                # --- ML Optimization: Initial Fit ---
                X_init = st.session_state.vectorizer.fit_transform(df_labeled_init["text"].tolist())
                # Store the vocabulary and the number of features learned from the initial fit
                st.session_state.vectorizer_vocabulary = st.session_state.vectorizer.vocabulary_
                st.session_state.expected_features_count = X_init.shape[1]

                y_init = st.session_state.le.transform(df_labeled_init["label"].tolist()) # VADER labels for initial training

                if len(np.unique(y_init)) > 1:
                    classes_unique_init = np.unique(y_init)
                    weights_init = compute_class_weight('balanced', classes=classes_unique_init, y=y_init)
                    class_weights_dict_init = dict(zip(classes_unique_init, weights_init))
                    sample_weights_for_fit_init = np.array([class_weights_dict_init[c] for c in y_init])
                else:
                    sample_weights_for_fit_init = None
                    st.warning("Only one class found in initial batch. Cannot compute balanced class weights.")

                if X_init.shape[0] > 0 and X_init.shape[1] > 0:
                    st.session_state.classifier.partial_fit(
                        X_init, y_init,
                        classes=st.session_state.all_possible_classes_array,
                        sample_weight=sample_weights_for_fit_init
                    )
                else:
                    st.warning("No features extracted from initial batch for classifier. Skipping initial fit.")
                    
                st.session_state.is_first_run_complete = True
                st.session_state.current_processed_count = len(initial_messages) 

                current_latest_sentiments = {}

                df_init.rename(columns={'cleaned_body': 'text'}, inplace=True)                
                df_init = df_init.merge(df_labeled_init[['text', 'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']], 
                                        on='text', how='left') 
                

                # Perform SGD prediction on the initial batch for cumulative storage
                if not df_init.empty and hasattr(st.session_state.classifier, 'coef_'):
                    X_init_predict = st.session_state.vectorizer.transform(df_init['text'].tolist()) # Use 'text'
                    
                    # Ensure classifier has been fitted with at least 2 classes
                    if hasattr(st.session_state.classifier, 'classes_') and len(st.session_state.classifier.classes_) > 1:
                        sgd_probs_init = st.session_state.classifier.predict_proba(X_init_predict)
                        sgd_predictions_init = st.session_state.classifier.predict(X_init_predict)
                        
                        df_init['sgd_label'] = st.session_state.le.inverse_transform(sgd_predictions_init)
                        df_init['sgd_confidence'] = np.max(sgd_probs_init, axis=1)
                    else:
                        st.warning("SGD Classifier not fully fitted (needs at least 2 classes) on initial batch. Defaulting to neutral for SGD labels.")
                        df_init['sgd_label'] = "neutral"
                        df_init['sgd_confidence'] = 0.5
                else:
                    df_init['sgd_label'] = "neutral"
                    df_init['sgd_confidence'] = 0.5
                
                # Append initial batch to cumulative DataFrame
                st.session_state.cumulative_comments_df = pd.concat([st.session_state.cumulative_comments_df, df_init], ignore_index=True)


                for kw in keywords_to_track:
                    if not kw: 
                        continue
                    # Use df_init for keyword filtering as it contains all original data and newly added SGD labels
                    keyword_comments_init_df = df_init[
                        df_init['text'].str.contains(kw, case=False, na=False)
                    ].copy()
                    
                    leading_confidence = 0.5
                    leading_label = "neutral"
                    comment_count = len(keyword_comments_init_df) 

                    if not keyword_comments_init_df.empty and 'sgd_label' in keyword_comments_init_df.columns:
                        # Calculate average sentiment from SGD predictions for the trend chart
                        sentiment_counts_kw = keyword_comments_init_df['sgd_label'].value_counts(normalize=True)
                        if not sentiment_counts_kw.empty:
                            leading_label = sentiment_counts_kw.idxmax()
                            leading_confidence = sentiment_counts_kw.max()
                            if not keyword_comments_init_df[keyword_comments_init_df['sgd_label'] == leading_label].empty:
                                leading_confidence = keyword_comments_init_df[keyword_comments_init_df['sgd_label'] == leading_label]['sgd_confidence'].mean()
                            else:
                                leading_confidence = 0.5 # Default if no comments for the leading label

                    st.session_state.keyword_data[kw]['confidences'].append(leading_confidence)
                    st.session_state.keyword_data[kw]['timestamps'].append(current_batch_timestamp) 
                    st.session_state.keyword_data[kw]['labels'].append(leading_label)
                    st.session_state.keyword_data[kw]['counts'].append(comment_count) 
                    
                    current_latest_sentiments[kw] = {'confidence': leading_confidence, 'label': leading_label} 
                        
                # Pass column containers to update_display and new display_top_subreddits, with new layout
                with col_left: # Bar chart goes here
                    update_display(
                        st.session_state.keyword_data,
                        chart_placeholder,
                        comment_volume_placeholder, 
                        timestamp_display,
                        processed_display,
                        current_sentiments_display,
                        current_batch_timestamp,
                        st.session_state.current_processed_count,
                        current_latest_sentiments,
                        keywords_to_track
                    )
                with col_right: # Top subreddits go here
                    display_top_subreddits(st.session_state.cumulative_comments_df, keywords_to_track, subreddit_placeholder)
                
                # Pie charts go below the two columns
                display_sentiment_distribution(st.session_state.cumulative_comments_df, keywords_to_track, sentiment_distribution_placeholder)
                
            st.markdown("### Processing Live Stream...")
            
        df_batch = pd.DataFrame()
        df_labeled_batch = pd.DataFrame() 
        current_batch_timestamp = pd.to_datetime(time.time(), unit='s')
        current_latest_sentiments = {} 
        
        # --- Main Loop for Live Stream Processing ---
        while True: 
            st.session_state.loop_counter += 1 # loop_counter always increments
            
            # PROACTIVE CHECK: Ensure consumer is open before polling
            if st.session_state.kafka_consumer is None or st.session_state.kafka_consumer._closed:
                st.warning("Detected Kafka consumer is closed. Attempting to re-initialize.")
                st.session_state.kafka_consumer = get_kafka_consumer()
                if st.session_state.kafka_consumer is None:
                    st.error("Failed to re-initialize Kafka consumer. Please check your Kafka setup and restart if issues persist.")
                    time.sleep(5) # Wait before next retry attempt
                    continue # Skip to next iteration of while True

            try:
                batch_messages = fetch_kafka_messages(st.session_state.kafka_consumer, BATCH_SIZE)
                
                current_batch_timestamp = pd.to_datetime(time.time(), unit='s')

                if not batch_messages:
                    st.info(f"No new Kafka messages. Waiting for data...")
                    df_batch = pd.DataFrame() 
                    df_labeled_batch = pd.DataFrame() # No new data means empty batch for training/vader labeling
                    
                    for kw in keywords_to_track:
                        if kw and kw in st.session_state.keyword_data and st.session_state.keyword_data[kw]['confidences']:
                            current_latest_sentiments[kw] = {
                                'confidence': st.session_state.keyword_data[kw]['confidences'][-1],
                                'label': st.session_state.keyword_data[kw]['labels'][-1]
                            }
                        elif kw: 
                             current_latest_sentiments[kw] = {'confidence': 0.5, 'label': "neutral"}
                else: 
                    df_batch = pd.DataFrame(batch_messages)
                    st.session_state.current_processed_count += len(batch_messages)
                    
                    df_batch['created_utc'] = pd.to_datetime(df_batch['created_utc'], unit='s')
                    df_batch = df_batch.sort_values(by='created_utc').reset_index(drop=True)
                    current_batch_timestamp = df_batch['created_utc'].iloc[-1] 

                    texts_raw_batch = df_batch["cleaned_body"].tolist()
                    df_labeled_batch, skipped_batch = vader_label_texts(texts_raw_batch) # df_labeled_batch still stores VADER

                    df_batch_for_cumulative = df_batch.copy()
                    df_batch_for_cumulative.rename(columns={'cleaned_body': 'text'}, inplace=True)

                    df_batch_for_cumulative = df_batch_for_cumulative.merge(
                        df_labeled_batch.drop_duplicates(subset=['text']), on='text', how='left' # Removed suffixes
                    )
                    
                    # Ensure the vectorizer has the correct vocabulary before transforming
                    if st.session_state.vectorizer_vocabulary is not None and \
                       (st.session_state.vectorizer.vocabulary_ is None or \
                        len(st.session_state.vectorizer.vocabulary_) != st.session_state.expected_features_count):
                        
                        st.warning("CountVectorizer vocabulary mismatch detected. Attempting to restore vocabulary.")
                        # Recreate the vectorizer with the stored vocabulary to ensure consistency
                        st.session_state.vectorizer = CountVectorizer(
                            min_df=2, max_features=10000, vocabulary=st.session_state.vectorizer_vocabulary
                        )

                    # Perform SGD prediction on df_batch_for_cumulative (comments to be added to cumulative)
                    if not df_batch_for_cumulative.empty and hasattr(st.session_state.classifier, 'coef_'):
                        X_batch_predict = st.session_state.vectorizer.transform(df_batch_for_cumulative['text'].tolist()) # Use 'text'
                        
                        if hasattr(st.session_state.classifier, 'classes_') and len(st.session_state.classifier.classes_) > 1:
                            sgd_probs_batch = st.session_state.classifier.predict_proba(X_batch_predict)
                            sgd_predictions_batch = st.session_state.classifier.predict(X_batch_predict)
                            
                            df_batch_for_cumulative['sgd_label'] = st.session_state.le.inverse_transform(sgd_predictions_batch)
                            df_batch_for_cumulative['sgd_confidence'] = np.max(sgd_probs_batch, axis=1)
                        else:
                            st.warning("SGD Classifier not fully fitted yet (needs at least 2 classes) for current batch. Defaulting to neutral for SGD labels.")
                            df_batch_for_cumulative['sgd_label'] = "neutral"
                            df_batch_for_cumulative['sgd_confidence'] = 0.5
                    else:
                        df_batch_for_cumulative['sgd_label'] = "neutral"
                        df_batch_for_cumulative['sgd_confidence'] = 0.5

                    # Append to cumulative DataFrame
                    st.session_state.cumulative_comments_df = pd.concat([st.session_state.cumulative_comments_df, df_batch_for_cumulative], ignore_index=True)
                    
                    if df_labeled_batch.empty: # This means VADER couldn't label anything in the batch
                        st.warning(f"No sufficient labeled data from VADER in current batch. Skipping classifier update and using previous sentiments for trend.")
                        for kw in keywords_to_track:
                            if kw and kw in st.session_state.keyword_data and st.session_state.keyword_data[kw]['confidences']:
                                # Use the last known values for trend if no new data to train on
                                st.session_state.keyword_data[kw]['confidences'].append(st.session_state.keyword_data[kw]['confidences'][-1])
                                st.session_state.keyword_data[kw]['timestamps'].append(current_batch_timestamp) 
                                st.session_state.keyword_data[kw]['labels'].append(st.session_state.keyword_data[kw]['labels'][-1])
                            elif kw: 
                                st.session_state.keyword_data[kw]['confidences'].append(0.5)
                                st.session_state.keyword_data[kw]['timestamps'].append(current_batch_timestamp)
                                st.session_state.keyword_data[kw]['labels'].append("neutral")
                        
                        current_latest_sentiments = {kw: {'confidence': st.session_state.keyword_data[kw]['confidences'][-1], 
                                                           'label': st.session_state.keyword_data[kw]['labels'][-1]} 
                                                     for kw in keywords_to_track if kw and kw in st.session_state.keyword_data}
                        
                    else: 
                        X_batch_new = st.session_state.vectorizer.transform(df_labeled_batch["text"].tolist())
                        y_batch_new = st.session_state.le.transform(df_labeled_batch["label"].tolist()) # Still using VADER for training

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
                        else:
                            pass 

                        current_latest_sentiments = {} 
                        for kw in keywords_to_track:
                            if not kw: 
                                continue
                            
                            # Filter the newly processed batch for comments containing the keyword and ensure they have SGD predictions
                            keyword_comments_current_batch_sgd = df_batch_for_cumulative[
                                df_batch_for_cumulative['text'].str.contains(kw, case=False, na=False) # Use 'text'
                            ].copy() 
                            
                            leading_confidence = None
                            leading_label = None

                            if not keyword_comments_current_batch_sgd.empty and 'sgd_label' in keyword_comments_current_batch_sgd.columns:
                                # For the trend chart, calculate average sentiment/confidence from the SGD predictions of the batch
                                sgd_labels_kw_batch = keyword_comments_current_batch_sgd['sgd_label']
                                
                                if not sgd_labels_kw_batch.empty:
                                    # Get the most frequent sentiment in this keyword batch
                                    leading_label = sgd_labels_kw_batch.mode()[0] 
                                    
                                    # Average the confidence scores for the dominant sentiment within this batch
                                    confidences_for_leading_label = keyword_comments_current_batch_sgd[
                                        keyword_comments_current_batch_sgd['sgd_label'] == leading_label
                                    ]['sgd_confidence']
                                    
                                    if not confidences_for_leading_label.empty:
                                        leading_confidence = confidences_for_leading_label.mean()
                                    else:
                                        leading_confidence = 0.5 # Default if no comments for the leading label
                                else:
                                    leading_confidence = 0.5
                                    leading_label = "neutral"
                            else: 
                                if kw and kw in st.session_state.keyword_data and st.session_state.keyword_data[kw]['confidences']:
                                    leading_confidence = st.session_state.keyword_data[kw]['confidences'][-1]
                                    leading_label = st.session_state.keyword_data[kw]['labels'][-1]
                                else: 
                                    leading_confidence = 0.5
                                    leading_label = "neutral"
                                pass 

                            st.session_state.keyword_data[kw]['confidences'].append(leading_confidence)
                            st.session_state.keyword_data[kw]['timestamps'].append(current_batch_timestamp) 
                            st.session_state.keyword_data[kw]['labels'].append(leading_label)
                            
                            current_latest_sentiments[kw] = {'confidence': leading_confidence, 'label': leading_label}
                
                # Pass column containers to update_display and display_sentiment_distribution with new layout
                with col_left:
                    update_display(
                        st.session_state.keyword_data,
                        chart_placeholder,
                        comment_volume_placeholder, 
                        timestamp_display, 
                        processed_display,
                        current_sentiments_display,
                        current_batch_timestamp, 
                        st.session_state.current_processed_count,
                        current_latest_sentiments,
                        keywords_to_track 
                    )
                with col_right:
                    display_top_subreddits(st.session_state.cumulative_comments_df, keywords_to_track, subreddit_placeholder)
                
                display_sentiment_distribution(st.session_state.cumulative_comments_df, keywords_to_track, sentiment_distribution_placeholder)

                
                time.sleep(0.1) 
                
            except KafkaError as e:
                st.error(f"Kafka error encountered: {e}. Please ensure broker is running and topic `{KAFKA_TOPIC}` exists. Retrying on next cycle...")
                time.sleep(5) 
                
                for kw in keywords_to_track:
                    if kw and kw in st.session_state.keyword_data and st.session_state.keyword_data[kw]['confidences']:
                        current_latest_sentiments[kw] = {
                            'confidence': st.session_state.keyword_data[kw]['confidences'][-1],
                            'label': st.session_state.keyword_data[kw]['labels'][-1]
                        }
                    elif kw:
                        current_latest_sentiments[kw] = {'confidence': 0.5, 'label': "neutral"}
                
                # Still pass column containers on error, with new layout
                with col_left:
                    update_display(
                        st.session_state.keyword_data, chart_placeholder, comment_volume_placeholder, timestamp_display, processed_display,
                        current_sentiments_display, current_batch_timestamp if 'current_batch_timestamp' in locals() else pd.to_datetime(time.time(), unit='s'),
                        st.session_state.current_processed_count, current_latest_sentiments, keywords_to_track
                    )
                with col_right:
                    display_top_subreddits(st.session_state.cumulative_comments_df, keywords_to_track, subreddit_placeholder)
                
                display_sentiment_distribution(st.session_state.cumulative_comments_df, keywords_to_track, sentiment_distribution_placeholder)
                
                time.sleep(0.1) 
                
                continue 
            except Exception as e:
                st.error(f"An unexpected error occurred during stream processing: {str(e)}. Attempting to continue.")
                
                for kw in keywords_to_track:
                    if kw and kw in st.session_state.keyword_data and st.session_state.keyword_data[kw]['confidences']:
                        st.session_state.keyword_data[kw]['confidences'].append(st.session_state.keyword_data[kw]['confidences'][-1])
                        st.session_state.keyword_data[kw]['timestamps'].append(current_batch_timestamp if 'current_batch_timestamp' in locals() else pd.to_datetime(time.time(), unit='s'))
                        st.session_state.keyword_data[kw]['labels'].append(st.session_state.keyword_data[kw]['labels'][-1])
                    elif kw:
                        st.session_state.keyword_data[kw]['confidences'].append(0.5)
                        st.session_state.keyword_data[kw]['timestamps'].append(current_batch_timestamp if 'current_batch_timestamp' in locals() else pd.to_datetime(time.time(), unit='s'))
                        st.session_state.keyword_data[kw]['labels'].append("neutral")
                    current_latest_sentiments[kw] = {'confidence': st.session_state.keyword_data[kw]['confidences'][-1],
                                                      'label': st.session_state.keyword_data[kw]['labels'][-1]}
                
                # Still pass column containers on error, with new layout
                with col_left:
                    update_display(
                        st.session_state.keyword_data, chart_placeholder, comment_volume_placeholder, timestamp_display, processed_display,
                        current_sentiments_display, current_batch_timestamp if 'current_batch_timestamp' in locals() else pd.to_datetime(time.time(), unit='s'),
                        st.session_state.current_processed_count, current_latest_sentiments, keywords_to_track
                    )
                with col_right:
                    display_top_subreddits(st.session_state.cumulative_comments_df, keywords_to_track, subreddit_placeholder)
                
                display_sentiment_distribution(st.session_state.cumulative_comments_df, keywords_to_track, sentiment_distribution_placeholder)
                
                time.sleep(0.1)
                continue 

if __name__ == "__main__":
    main()