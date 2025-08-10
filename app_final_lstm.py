import pandas as pd
import numpy as np
import streamlit as st
import time
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from kafka import KafkaConsumer
from kafka.errors import KafkaError

# --- LSTM Model and Preprocessing Components ---
import torch
import torch.nn as nn
import torch.optim as optim
from afinn import Afinn
from nltk.tokenize import word_tokenize
from collections import Counter
import emoji

# Ensure NLTK 'punkt' is downloaded
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- Constants ---
INIT_ROWS = 50_000
# Increased batch size for faster data consumption
BATCH_SIZE = 10_000 

# --- Kafka Configuration ---
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
KAFKA_TOPIC = 'reddit_comments_processed'
# Changed group ID to ensure Kafka starts from the earliest offset on app restart
KAFKA_CONSUMER_GROUP_ID = 'streamlit_lstm_sentiment_group_v3' 

# --- Define keys expected in Kafka JSON messages ---
ALL_KAFKA_MESSAGE_KEYS = ['cleaned_body', 'original_score', 'subreddit', 'created_utc']

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Live LSTM Sentiment Tracker (Dual Keyword)",
    page_icon="🧠",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .positive { color: #2ecc71; } /* Green */
    .neutral { color: #f39c12; } /* Orange */
    .negative { color: #e74c3c; } /* Red */
    .keyword-sentiment-box {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        display: inline-block;
        min-width: 48%;
        box-sizing: border-box;
    }
</style>
""", unsafe_allow_html=True)

# ========== LSTM Preprocessing ==========
def preprocess_text(text):
    text = emoji.demojize(str(text), delimiters=(" ", " "))
    return word_tokenize(text.lower())

def afinn_label_texts(texts):
    af = Afinn()
    data = []
    for t in texts:
        score = af.score(t)
        if score > 0: label = 1.0
        elif score < 0: label = 0.0
        else: continue
        tokens = preprocess_text(t)
        if tokens: data.append((t, tokens, label))
    return pd.DataFrame(data, columns=["text", "tokens", "label"])

def build_vocab(token_lists, min_freq=2):
    counter = Counter(tok for tokens in token_lists for tok in tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for tok, count in counter.items():
        if count >= min_freq: vocab[tok] = len(vocab)
    return vocab

def encode(tokens, vocab, max_len=50):
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    return ids[:max_len] + [vocab["<PAD>"]] * max(0, max_len - len(ids))

# ========== LSTM Model ==========
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        x = self.dropout(hidden[-1])
        return self.fc(x).squeeze(1)

# ========== Model Training and Prediction ==========
@st.cache_resource(show_spinner="Training Initial LSTM Model...")
def train_initial_model(initial_data_df):
    st.info("Labeling initial texts with AFINN for training...")
    df_labeled = afinn_label_texts(initial_data_df["text"].dropna().astype(str))
    if df_labeled.empty or len(df_labeled['label'].unique()) < 2:
        st.error("Not enough varied data for initial training. Please check the data source.")
        st.stop()

    st.info("Building vocabulary...")
    vocab = build_vocab(df_labeled['tokens'])
    model = SimpleLSTM(vocab_size=len(vocab))

    df_labeled['input'] = df_labeled['tokens'].apply(lambda x: encode(x, vocab))
    X = torch.tensor(df_labeled['input'].tolist(), dtype=torch.long)
    y = torch.tensor(df_labeled['label'].tolist(), dtype=torch.float)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    st.info("Starting model training...")
    model.train()
    perm = torch.randperm(len(X))
    for i in range(0, len(X), 128):
        idx = perm[i:i+128]
        x_batch, y_batch = X[idx], y[idx]
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    st.success("✅ Initial LSTM model trained.")
    return model, vocab

def predict_sentiment_batch(texts, model, vocab):
    model.eval()
    results = []
    with torch.no_grad():
        for text in texts:
            tokens = preprocess_text(text)
            ids = encode(tokens, vocab)
            x = torch.tensor([ids], dtype=torch.long)
            prob = torch.sigmoid(model(x)).item()
            if prob >= 0.55: label = "positive"
            elif prob <= 0.45: label = "negative"
            else: label = "neutral"
            confidence = abs(prob - 0.5) * 2
            results.append((label, confidence))
    return results

# ========== Kafka and Streamlit Components ==========
@st.cache_resource(show_spinner=False)
def get_kafka_consumer():
    st.info("Connecting to Kafka...")
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=KAFKA_CONSUMER_GROUP_ID,
            auto_offset_reset='earliest',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        st.success(f"Successfully subscribed to Kafka topic: `{KAFKA_TOPIC}`")
        return consumer
    except KafkaError as e:
        st.error(f"Failed to connect to Kafka: {e}.")
        st.stop()

def fetch_kafka_messages(consumer: KafkaConsumer, num_messages: int, timeout_ms: int = 3000):
    messages_list = []
    try:
        message_batch = consumer.poll(timeout_ms=timeout_ms, max_records=num_messages)
        for tp, records in message_batch.items():
            for msg in records:
                if all(key in msg.value for key in ALL_KAFKA_MESSAGE_KEYS):
                    messages_list.append(msg.value)
    except KafkaError as e:
        st.error(f"Kafka error during message fetch: {e}")
    return messages_list

# ========== Visualization Functions ==========
def display_sentiment_distribution(all_comments_df, keywords_to_track, placeholder):
    if all_comments_df.empty:
        placeholder.empty()
        return

    charts = []
    for kw in keywords_to_track:
        if not kw: continue
        kw_df = all_comments_df[all_comments_df['text'].str.contains(kw, case=False, na=False)]
        
        if not kw_df.empty:
            counts = kw_df['lstm_label'].value_counts(normalize=True).reset_index()
            counts.columns = ['Sentiment', 'Percentage']
            counts['Percentage'] *= 100
            colors = {"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"}
            pie = px.pie(
                counts, values='Percentage', names='Sentiment',
                title=f'Cumulative Sentiment for "{kw}"', color='Sentiment',
                color_discrete_map=colors, hole=0.3
            )
            pie.update_traces(textinfo='percent+label')
            pie.update_layout(margin=dict(l=20, r=20, t=60, b=20))
            charts.append(pie)

    if charts:
        with placeholder.container():
            st.markdown("---"); st.markdown("### Cumulative Sentiment Breakdown")
            cols = st.columns(len(charts))
            for i, chart in enumerate(charts):
                # **FIX**: Use a unique key with keyword, index, and loop counter
                key = f"pie_chart_{keywords_to_track[i]}_{i}_{st.session_state.loop_counter}"
                cols[i].plotly_chart(chart, use_container_width=True, key=key)

def update_display(keyword_data, chart_placeholder, volume_placeholder, ts_display, proc_display, sent_display, current_ts, proc_count, latest_sentiments, keywords):
    ts_display.markdown(f"`{current_ts.strftime('%Y-%m-%d %H:%M:%S')}`")
    proc_display.markdown(f"`{proc_count:,}`")
    
    fig = go.Figure()
    colors = {"positive": "lightgreen", "neutral": "gold", "negative": "red"}
    lines = {keywords[0]: 'darkblue', keywords[1]: 'gray'} if len(keywords) > 1 else {}

    for kw, data in keyword_data.items():
        if not data['timestamps']: continue
        df = pd.DataFrame(data)
        df['MarkerColor'] = df['labels'].map(colors)
        df['Confidence Score (%)'] = df['confidences'] * 100
        
        fig.add_trace(go.Scatter(
            x=df["timestamps"], y=df["Confidence Score (%)"], mode="markers+lines",
            marker=dict(color=df["MarkerColor"], size=10, line=dict(width=1, color='DarkSlateGrey')),
            line=dict(color=lines.get(kw, 'grey'), width=2),
            name=f"{kw} Confidence", customdata=df["labels"],
            hovertemplate="<b>%{customdata}</b><br>Confidence: %{y:.1f}%<extra></extra>"
        ))
    fig.update_layout(title="Sentiment Confidence Trend (Batch Average)", yaxis_title="Avg. Confidence (%)", yaxis_range=[0, 105])
    # **FIX**: Use a unique key with loop counter
    chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"trend_chart_{st.session_state.loop_counter}")

    bar_fig = go.Figure()
    if 'cumulative_comments_df' in st.session_state and not st.session_state.cumulative_comments_df.empty:
        counts = [{'Keyword': kw, 'Total Comments': st.session_state.cumulative_comments_df['text'].str.contains(kw, case=False, na=False).sum()} for kw in keywords if kw]
        if counts:
            df_counts = pd.DataFrame(counts)
            bar_fig.add_trace(go.Bar(x=df_counts["Keyword"], y=df_counts["Total Comments"], marker_color=[lines.get(k, 'lightgrey') for k in df_counts["Keyword"]], text=df_counts["Total Comments"], textposition='outside'))
    bar_fig.update_layout(title="Cumulative Comment Volume Per Keyword", showlegend=False)
    # **FIX**: Use a unique key with loop counter
    volume_placeholder.plotly_chart(bar_fig, use_container_width=True, key=f"volume_chart_{st.session_state.loop_counter}")

    with sent_display.container():
        st.markdown("<h4>Current Keyword Sentiments:</h4>")
        cols = st.columns(len(latest_sentiments))
        for idx, (kw, info) in enumerate(latest_sentiments.items()):
            cols[idx].markdown(f"<div class='keyword-sentiment-box'><b>{kw}:</b> <span class='{info['label']}'>{info['label'].capitalize()}</span> (Confidence: {info['confidence']:.1%})</div>", unsafe_allow_html=True)

def display_top_subreddits(df, keywords, placeholder):
    if df.empty: return
    with placeholder.container():
        st.markdown("### Top Subreddits by Keyword")
        cols = st.columns(len(keywords))
        for idx, kw in enumerate(keywords):
            if not kw: continue
            with cols[idx]:
                st.subheader(f'For "{kw}"')
                kw_df = df[df['text'].str.contains(kw, case=False, na=False)]
                if not kw_df.empty:
                    top_subs = kw_df['subreddit'].value_counts().nlargest(5).reset_index()
                    top_subs.columns = ['Subreddit', 'Count']
                    # The dataframe widget does not need a key if its data changes
                    st.dataframe(top_subs, hide_index=True, use_container_width=True)
                else:
                    st.info("No comments yet.")

# ========== Main Application Logic ==========
def main():
    st.title("🧠 Live LSTM Sentiment Tracker (Dual Keyword)")
    with st.sidebar:
        st.header("⚙️ Controls")
        keyword1 = st.text_input("Enter Keyword 1:", value="AI")
        keyword2 = st.text_input("Enter Keyword 2:", value="data")
        run_button = st.button("Start Analysis", type="primary")
        if st.button("Reset"): st.session_state.clear(); st.rerun()
        st.markdown("---")
        st.markdown("**Last Record Timestamp:**"); ts_display = st.empty()
        st.markdown("**Processed Comments:**"); proc_display = st.empty()

    chart_placeholder = st.empty(); sent_display = st.empty()
    st.markdown("---")
    col_left, col_right = st.columns(2)
    with col_left: volume_placeholder = st.empty()
    with col_right: subreddit_placeholder = st.empty()
    dist_placeholder = st.empty()

    if 'model' not in st.session_state:
        st.session_state.keyword_data = {}
        st.session_state.processed_count = 0
        st.session_state.is_first_run = True
        st.session_state.cumulative_comments_df = pd.DataFrame(columns=['text', 'subreddit', 'created_utc', 'lstm_label', 'lstm_confidence'])
        # **FIX**: Initialize the loop counter
        st.session_state.loop_counter = 0

    keywords = [keyword1, keyword2]
    for kw in keywords:
        if kw and kw not in st.session_state.keyword_data:
            st.session_state.keyword_data[kw] = {'timestamps': [], 'confidences': [], 'labels': []}

    if not run_button: st.stop()
    
    if st.session_state.is_first_run:
        with st.spinner(f"Fetching initial {INIT_ROWS:,} comments for model training..."):
            consumer = get_kafka_consumer()
            initial_messages = fetch_kafka_messages(consumer, INIT_ROWS, timeout_ms=10000)
        if not initial_messages:
            st.error("❌ No messages received from Kafka for initial training."); st.stop()
        
        df_init = pd.DataFrame(initial_messages)
        df_init.rename(columns={'cleaned_body': 'text'}, inplace=True)
        
        st.session_state.model, st.session_state.vocab = train_initial_model(df_init)
        st.session_state.kafka_consumer = consumer
        st.session_state.is_first_run = False
        st.success("Initialization complete. Starting live processing...")

    while True:
        # **FIX**: Increment the loop counter at the start of each iteration
        st.session_state.loop_counter += 1
        try:
            messages = fetch_kafka_messages(st.session_state.kafka_consumer, BATCH_SIZE)
            current_ts = datetime.now()
            latest_sentiments = {}
            df_batch = pd.DataFrame() # Initialize empty df

            if messages:
                df_batch = pd.DataFrame(messages)
                df_batch.rename(columns={'cleaned_body': 'text'}, inplace=True)
                st.session_state.processed_count += len(df_batch)
                
                texts = df_batch['text'].dropna().astype(str).tolist()
                if texts:
                    preds = predict_sentiment_batch(texts, st.session_state.model, st.session_state.vocab)
                    df_preds = pd.DataFrame(preds, columns=['lstm_label', 'lstm_confidence'])
                    df_batch = pd.concat([df_batch.reset_index(drop=True), df_preds], axis=1)
                    st.session_state.cumulative_comments_df = pd.concat([st.session_state.cumulative_comments_df, df_batch], ignore_index=True)
                
                if 'created_utc' in df_batch and not df_batch['created_utc'].empty:
                    current_ts = pd.to_datetime(df_batch['created_utc'].max(), unit='s')

            for kw in keywords:
                if not kw: continue
                kw_df = df_batch[df_batch['text'].str.contains(kw, case=False, na=False)] if not df_batch.empty else pd.DataFrame()
                
                if not kw_df.empty:
                    conf, label = kw_df['lstm_confidence'].mean(), kw_df['lstm_label'].mode()[0]
                else: # Carry over last known sentiment if no new comments for this keyword
                    conf = st.session_state.keyword_data[kw]['confidences'][-1] if st.session_state.keyword_data[kw]['confidences'] else 0.5
                    label = st.session_state.keyword_data[kw]['labels'][-1] if st.session_state.keyword_data[kw]['labels'] else "neutral"

                st.session_state.keyword_data[kw]['confidences'].append(conf)
                st.session_state.keyword_data[kw]['timestamps'].append(current_ts)
                st.session_state.keyword_data[kw]['labels'].append(label)
                latest_sentiments[kw] = {'confidence': conf, 'label': label}
            
            update_display(st.session_state.keyword_data, chart_placeholder, volume_placeholder, ts_display, proc_display, sent_display, current_ts, st.session_state.processed_count, latest_sentiments, keywords)
            display_top_subreddits(st.session_state.cumulative_comments_df, keywords, subreddit_placeholder)
            display_sentiment_distribution(st.session_state.cumulative_comments_df, keywords, dist_placeholder)
            
            # Reduced sleep time for faster polling
            time.sleep(1)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            time.sleep(5)
            continue

if __name__ == "__main__":
    main()