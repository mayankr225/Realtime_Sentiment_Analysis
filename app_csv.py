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

# --- Constants ---
FILE_PATH = "cleaned_comments_full.csv" # Make sure this file exists in your project directory
TOTAL_ROWS = 47_741_033 # Your total number of rows
INIT_ROWS = 100_000 # Your chosen initial batch size
BATCH_SIZE = 50_000 # Your chosen batch size
MAX_BATCHES_DISPLAY = 150 # Limits how many batches are processed in the demo

# --- Define all column names in the correct order for headerless reads ---
# This list MUST match the exact headers in your cleaned_comments_full.csv
ALL_CSV_COLUMNS = ['cleaned_text', 'score', 'subreddit', 'created_utc']

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
def initialize_vectorizer_and_label_encoder():
    """Initializes CountVectorizer and LabelEncoder once."""
    vectorizer = CountVectorizer(min_df=2, max_features=10000) # Increased max_features for better vocab
    le = LabelEncoder()
    all_classes = ["negative", "neutral", "positive"]
    le.fit(all_classes) # Fit once on all possible classes
    return vectorizer, le

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
        
        data.append((processed_text, label))
    return pd.DataFrame(data, columns=["text", "label"]), skipped_count


# --- Main Application Logic ---
def main():
    st.title("📊 Live Sentiment Tracker (Dual Keyword Comparison)")
    st.markdown("Real-time sentiment analysis on large datasets, tracking sentiment around **two different keywords** and comparing their trends.")
    
    with st.sidebar:
        st.header("⚙️ Controls")
        keyword1 = st.text_input("Enter Keyword 1:", value="AI", placeholder="e.g., Bitcoin")
        keyword2 = st.text_input("Enter Keyword 2:", value="data", placeholder="e.g., climate")

        col_buttons = st.columns(2)
        with col_buttons[0]:
            run_button = st.button("Start Analysis", type="primary")
        with col_buttons[1]:
            reset_button = st.button("Reset", type="secondary")

        st.markdown("---")
        st.markdown("**Last Record Timestamp:**")
        timestamp_display = st.empty() # Changed from batch_display
        st.markdown("**Processed Comments:**")
        processed_display = st.empty()
    
    # --- Session State Initialization ---
    if 'vectorizer' not in st.session_state or 'le' not in st.session_state:
        st.session_state.vectorizer, st.session_state.le = initialize_vectorizer_and_label_encoder()
        st.session_state.classifier = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
        st.session_state.is_first_run_complete = False
        st.session_state.all_texts_seen = []
        st.session_state.all_labels_seen = []
        
        # Dictionary to hold trend data for multiple keywords
        st.session_state.keyword_data = {} 
        
        st.session_state.current_processed_count = 0
        st.session_state.current_batch_idx = 0 # Still useful for internal logic/counting
        st.session_state.last_run_keywords = ["", ""] # Track both keywords from last run
        st.session_state.all_possible_classes_array = np.array(st.session_state.le.transform(["negative", "neutral", "positive"]))

    # List of keywords currently being tracked
    keywords_to_track = [keyword1, keyword2]

    # Reset button logic
    if reset_button:
        st.session_state.clear()
        st.rerun()

    # --- UI Placeholders for dynamic updates ---
    progress_bar = st.progress(0)
    chart_placeholder = st.empty()
    current_sentiments_display = st.empty() # For displaying both keywords' current sentiment

    # Initial screen message
    if not run_button and not st.session_state.is_first_run_complete:
        with st.container():
            st.info("💡 Enter two keywords and click 'Start Analysis' to begin tracking and comparing sentiment trends!")
            st.write("The application will train a sentiment model on the overall data, then analyze comments containing each keyword to show their individual sentiment trends.")
        return

    # --- File Existence Check ---
    if not os.path.exists(FILE_PATH):
        st.error(f"❌ Error: Data file '{FILE_PATH}' not found. Please ensure it's in the same directory as the script.")
        return

    # --- Check if keywords have changed or if it's a fresh run ---
    if run_button or st.session_state.last_run_keywords != keywords_to_track:
        if st.session_state.last_run_keywords != keywords_to_track:
            # Keywords changed, so reset all keyword-specific data
            st.session_state.keyword_data = {}
            st.session_state.is_first_run_complete = False # Re-run initial batch if keywords changed
            st.session_state.all_texts_seen = [] # Clear training data to retrain model if needed
            st.session_state.all_labels_seen = [] # Clear training data to retrain model if needed
            st.session_state.current_processed_count = 0
            st.session_state.current_batch_idx = 0
            st.session_state.classifier = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42) # Reinitialize classifier
            st.session_state.vectorizer, st.session_state.le = initialize_vectorizer_and_label_encoder() # Reinitialize vectorizer/le
            st.warning("Calculating Sentiments.......")

        st.session_state.last_run_keywords = keywords_to_track[:] # Store current keywords for next check

        # Ensure keyword data structures exist for new keywords, initialize with 'timestamps'
        for kw in keywords_to_track:
            if kw not in st.session_state.keyword_data:
                st.session_state.keyword_data[kw] = {'timestamps': [], 'confidences': [], 'labels': []}


    # --- Analysis Logic ---
    if run_button or st.session_state.is_first_run_complete:
        
        # Handle initial batch differently for vectorizer fit and initial classifier fit
        if not st.session_state.is_first_run_complete:
            with st.spinner(f'Initializing vectorizer and training model with first {INIT_ROWS} comments...'):
                try:
                    df_init = pd.read_csv(FILE_PATH, nrows=INIT_ROWS)
                except Exception as e:
                    st.error(f"Error loading initial data: {e}")
                    return

                # Get the timestamp of the last record in the initial batch
                if not df_init.empty:
                    last_utc_init = df_init['created_utc'].iloc[-1]
                    current_batch_timestamp = pd.to_datetime(last_utc_init, unit='s')
                else:
                    current_batch_timestamp = pd.to_datetime(time.time(), unit='s') # Fallback to current time
                    st.warning("Initial DataFrame is empty, using current time as timestamp.")


                texts_init_raw = df_init["cleaned_text"].tolist()
                df_labeled_init, skipped_init = vader_label_texts(texts_init_raw)
                
                if df_labeled_init.empty:
                    st.error("❌ No sufficient labeled data found in the initial batch for model training. Please check your data or VADER labeling logic.")
                    return
                
                st.session_state.all_texts_seen.extend(df_labeled_init["text"].tolist())
                st.session_state.all_labels_seen.extend(df_labeled_init["label"].tolist())

                # Fit the vectorizer only once on the initial, larger dataset
                X_cumulative = st.session_state.vectorizer.fit_transform(st.session_state.all_texts_seen)
                y_cumulative = st.session_state.le.transform(st.session_state.all_labels_seen)

                if len(np.unique(y_cumulative)) > 1:
                    classes_unique = np.unique(y_cumulative)
                    weights = compute_class_weight('balanced', classes=classes_unique, y=y_cumulative)
                    class_weights_dict = dict(zip(classes_unique, weights))
                else:
                    class_weights_dict = None

                if X_cumulative.shape[0] > 0 and X_cumulative.shape[1] > 0:
                    sample_weights_for_fit = None
                    if class_weights_dict is not None:
                        sample_weights_for_fit = np.array([class_weights_dict[c] for c in y_cumulative])

                    st.session_state.classifier.partial_fit(
                        X_cumulative, y_cumulative,
                        classes=st.session_state.all_possible_classes_array,
                        sample_weight=sample_weights_for_fit
                    )
                else:
                    st.warning("No features extracted from initial batch for classifier. Skipping initial fit.")
                    
                st.session_state.is_first_run_complete = True
                st.session_state.current_processed_count = INIT_ROWS
                st.session_state.current_batch_idx = 0 # Keep batch_idx for internal count

                # --- Get sentiment for comments containing EACH keyword in the initial batch ---
                current_latest_sentiments = {}
                for kw in keywords_to_track:
                    keyword_comments_init_df = df_labeled_init[
                        df_labeled_init['text'].str.contains(kw, case=False, na=False)
                    ]

                    leading_confidence = 0.5
                    leading_label = "neutral"

                    if not keyword_comments_init_df.empty and hasattr(st.session_state.classifier, 'coef_'):
                        X_keyword_comments_init = st.session_state.vectorizer.transform(keyword_comments_init_df['text'].tolist())
                        keyword_comment_probs_init = st.session_state.classifier.predict_proba(X_keyword_comments_init)
                        avg_probs_init = np.mean(keyword_comment_probs_init, axis=0)

                        leading_idx = np.argmax(avg_probs_init)
                        leading_confidence = avg_probs_init[leading_idx]
                        leading_label = st.session_state.le.inverse_transform([leading_idx])[0]
                    else:
                        st.warning(f"No comments found containing '{kw}' in initial batch, or classifier not fitted. Defaulting to neutral for {kw}.")
                    
                    st.session_state.keyword_data[kw]['confidences'].append(leading_confidence)
                    st.session_state.keyword_data[kw]['timestamps'].append(current_batch_timestamp) # Use the batch timestamp
                    st.session_state.keyword_data[kw]['labels'].append(leading_label)
                    
                    current_latest_sentiments[kw] = {'confidence': leading_confidence, 'label': leading_label}

                update_display(
                    st.session_state.keyword_data,
                    chart_placeholder,
                    timestamp_display, # Pass timestamp_display
                    processed_display,
                    current_sentiments_display, # Pass the placeholder for overall latest sentiments
                    current_batch_timestamp, # Pass the actual timestamp
                    st.session_state.current_processed_count,
                    current_latest_sentiments # Pass the dictionary of latest sentiments
                )
            
        # Process subsequent batches (partial_fit on cumulative data)
        for i in range(st.session_state.current_batch_idx + 1, MAX_BATCHES_DISPLAY + 1):
            start_row = INIT_ROWS + (i - 1) * BATCH_SIZE
            if start_row >= TOTAL_ROWS:
                st.info("End of file reached. Stopping analysis.")
                progress_bar.progress(1.0)
                break

            try:
                with st.spinner(f'Processing batch {i} (rows {start_row + 1} to {min(start_row + BATCH_SIZE, TOTAL_ROWS)})...'):
                    df_batch = pd.read_csv(
                        FILE_PATH,
                        skiprows=start_row + 1,
                        nrows=BATCH_SIZE,
                        header=None,
                        names=ALL_CSV_COLUMNS
                    )
                    
                    # Get the timestamp of the last record in the current batch
                    if not df_batch.empty:
                        last_utc_batch = df_batch['created_utc'].iloc[-1]
                        current_batch_timestamp = pd.to_datetime(last_utc_batch, unit='s')
                    else:
                        # If batch is empty, use the timestamp from the previous batch to keep continuity
                        first_keyword_in_data = next(iter(st.session_state.keyword_data.values()), None)
                        
                        timestamp_source_label = 'current time' # Default fallback
                        if first_keyword_in_data and first_keyword_in_data['timestamps']:
                            current_batch_timestamp = first_keyword_in_data['timestamps'][-1]
                            timestamp_source_label = "previous batch's timestamp" 
                        else:
                            current_batch_timestamp = pd.to_datetime(time.time(), unit='s')
                            # timestamp_source_label remains "current time"

                        # Use the pre-computed string in the f-string
                        st.warning(f"Batch {i} DataFrame is empty, using {timestamp_source_label} as timestamp.")


                    st.session_state.current_processed_count = start_row + BATCH_SIZE
                    st.session_state.current_batch_idx = i # Keep batch_idx for internal count

                    texts_raw_batch = df_batch["cleaned_text"].tolist()
                    df_labeled_batch, skipped_batch = vader_label_texts(texts_raw_batch)
                    
                    if df_labeled_batch.empty:
                        st.warning(f"No sufficient labeled data in batch {i}. Skipping classifier update and keyword sentiment calculation.")
                        # If no data in batch, carry over previous state for ALL keywords
                        for kw in keywords_to_track:
                             if st.session_state.keyword_data[kw]['confidences']:
                                st.session_state.keyword_data[kw]['confidences'].append(st.session_state.keyword_data[kw]['confidences'][-1])
                                st.session_state.keyword_data[kw]['timestamps'].append(current_batch_timestamp) # Use the batch timestamp
                                st.session_state.keyword_data[kw]['labels'].append(st.session_state.keyword_data[kw]['labels'][-1])
                             else:
                                st.session_state.keyword_data[kw]['confidences'].append(0.5)
                                st.session_state.keyword_data[kw]['timestamps'].append(current_batch_timestamp)
                                st.session_state.keyword_data[kw]['labels'].append("neutral")

                        # Pass current_latest_sentiments based on what was just appended
                        current_latest_sentiments = {kw: {'confidence': st.session_state.keyword_data[kw]['confidences'][-1], 
                                                           'label': st.session_state.keyword_data[kw]['labels'][-1]} 
                                                     for kw in keywords_to_track}

                        update_display(
                            st.session_state.keyword_data,
                            chart_placeholder,
                            timestamp_display,
                            processed_display,
                            current_sentiments_display,
                            current_batch_timestamp, # Pass the actual timestamp
                            st.session_state.current_processed_count,
                            current_latest_sentiments
                        )
                        progress_bar.progress(min(st.session_state.current_processed_count / TOTAL_ROWS, 1.0))
                        time.sleep(0.3)
                        continue
                        
                    # Add new data to the cumulative lists
                    st.session_state.all_texts_seen.extend(df_labeled_batch["text"].tolist())
                    st.session_state.all_labels_seen.extend(df_labeled_batch["label"].tolist())

                    # Reinitialize and fit the classifier on the *entire cumulative dataset* for stability
                    X_cumulative = st.session_state.vectorizer.transform(st.session_state.all_texts_seen)
                    y_cumulative = st.session_state.le.transform(st.session_state.all_labels_seen)

                    if len(np.unique(y_cumulative)) > 1:
                        classes_unique = np.unique(y_cumulative)
                        weights = compute_class_weight('balanced', classes=classes_unique, y=y_cumulative)
                        class_weights_dict = dict(zip(classes_unique, weights))
                    else:
                        class_weights_dict = None

                    current_model_for_batch = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
                    if X_cumulative.shape[0] > 0 and X_cumulative.shape[1] > 0:
                        sample_weights_for_fit = None
                        if class_weights_dict is not None:
                            sample_weights_for_fit = np.array([class_weights_dict[c] for c in y_cumulative])
                            
                        current_model_for_batch.partial_fit(
                            X_cumulative, y_cumulative,
                            classes=st.session_state.all_possible_classes_array,
                            sample_weight=sample_weights_for_fit
                        )
                        st.session_state.classifier = current_model_for_batch # Update the stored classifier
                    else:
                        st.warning(f"Batch {i} added data, but no features extracted for cumulative retraining. Skipping model update.")
                        # If model can't update, carry over previous state for ALL keywords
                        for kw in keywords_to_track:
                            if st.session_state.keyword_data[kw]['confidences']:
                                st.session_state.keyword_data[kw]['confidences'].append(st.session_state.keyword_data[kw]['confidences'][-1])
                                st.session_state.keyword_data[kw]['timestamps'].append(current_batch_timestamp)
                                st.session_state.keyword_data[kw]['labels'].append(st.session_state.keyword_data[kw]['labels'][-1])
                            else:
                                st.session_state.keyword_data[kw]['confidences'].append(0.5)
                                st.session_state.keyword_data[kw]['timestamps'].append(current_batch_timestamp)
                                st.session_state.keyword_data[kw]['labels'].append("neutral")
                        
                        # Pass current_latest_sentiments based on what was just appended
                        current_latest_sentiments = {kw: {'confidence': st.session_state.keyword_data[kw]['confidences'][-1], 
                                                           'label': st.session_state.keyword_data[kw]['labels'][-1]} 
                                                     for kw in keywords_to_track}
                        update_display(
                            st.session_state.keyword_data,
                            chart_placeholder,
                            timestamp_display,
                            processed_display,
                            current_sentiments_display,
                            current_batch_timestamp, # Pass the actual timestamp
                            st.session_state.current_processed_count,
                            current_latest_sentiments
                        )
                        progress_bar.progress(min(st.session_state.current_processed_count / TOTAL_ROWS, 1.0))
                        time.sleep(0.3)
                        continue


                    # --- Predict sentiment for comments containing EACH keyword ---
                    current_latest_sentiments = {} # Reset for this batch
                    for kw in keywords_to_track:
                        keyword_comments_batch_df = df_labeled_batch[
                            df_labeled_batch['text'].str.contains(kw, case=False, na=False)
                        ]

                        leading_confidence = None
                        leading_label = None

                        if not keyword_comments_batch_df.empty and hasattr(st.session_state.classifier, 'coef_'):
                            X_keyword_comments = st.session_state.vectorizer.transform(keyword_comments_batch_df['text'].tolist())

                            if hasattr(st.session_state.classifier, 'coef_'):
                                keyword_comment_probs = st.session_state.classifier.predict_proba(X_keyword_comments)
                                avg_probs = np.mean(keyword_comment_probs, axis=0)

                                leading_idx = np.argmax(avg_probs)
                                leading_confidence = avg_probs[leading_idx]
                                leading_label = st.session_state.le.inverse_transform([leading_idx])[0]
                            else:
                                # Classifier not fitted, use last or default
                                if st.session_state.keyword_data[kw]['confidences']:
                                    leading_confidence = st.session_state.keyword_data[kw]['confidences'][-1]
                                    leading_label = st.session_state.keyword_data[kw]['labels'][-1]
                                else:
                                    leading_confidence = 0.5
                                    leading_label = "neutral"
                                st.warning(f"Classifier not fitted for '{kw}' in batch {i}, defaulting to previous/neutral.")
                        else:
                            # If no comments in this batch contained the keyword, use the last known sentiment/confidence
                            if st.session_state.keyword_data[kw]['confidences']:
                                leading_confidence = st.session_state.keyword_data[kw]['confidences'][-1]
                                leading_label = st.session_state.keyword_data[kw]['labels'][-1]
                            else: # This should only happen for initial batch if keyword not found
                                leading_confidence = 0.5
                                leading_label = "neutral"
                            st.warning(f"No comments found containing '{kw}' in batch {i}. Displaying previous sentiment for '{kw}'.")
                        
                        st.session_state.keyword_data[kw]['confidences'].append(leading_confidence)
                        st.session_state.keyword_data[kw]['timestamps'].append(current_batch_timestamp) # Use the batch timestamp
                        st.session_state.keyword_data[kw]['labels'].append(leading_label)
                        
                        current_latest_sentiments[kw] = {'confidence': leading_confidence, 'label': leading_label}
                    
                    update_display(
                        st.session_state.keyword_data,
                        chart_placeholder,
                        timestamp_display,
                        processed_display,
                        current_sentiments_display,
                        current_batch_timestamp, # Pass the actual timestamp
                        st.session_state.current_processed_count,
                        current_latest_sentiments
                    )
                    
                    progress_bar.progress(min(st.session_state.current_processed_count / TOTAL_ROWS, 1.0))
                    time.sleep(0.3)
                    
            except pd.errors.EmptyDataError:
                st.info(f"End of file reached or no more data to read in batch {i}. Stopping analysis.")
                progress_bar.progress(1.0)
                break
            except Exception as e:
                st.error(f"An unexpected error occurred during batch {i}: {str(e)}. Skipping this batch and continuing.")
                # Append last known state to prevent chart breaking for ALL keywords
                current_latest_sentiments = {}
                for kw in keywords_to_track:
                    if st.session_state.keyword_data[kw]['confidences']:
                        st.session_state.keyword_data[kw]['confidences'].append(st.session_state.keyword_data[kw]['confidences'][-1])
                        st.session_state.keyword_data[kw]['timestamps'].append(current_batch_timestamp) # Use the current batch timestamp
                        st.session_state.keyword_data[kw]['labels'].append(st.session_state.keyword_data[kw]['labels'][-1])
                    else:
                        st.session_state.keyword_data[kw]['confidences'].append(0.5)
                        st.session_state.keyword_data[kw]['timestamps'].append(current_batch_timestamp)
                        st.session_state.keyword_data[kw]['labels'].append("neutral")
                    current_latest_sentiments[kw] = {'confidence': st.session_state.keyword_data[kw]['confidences'][-1],
                                                      'label': st.session_state.keyword_data[kw]['labels'][-1]}


                update_display(
                    st.session_state.keyword_data,
                    chart_placeholder,
                    timestamp_display,
                    processed_display,
                    current_sentiments_display,
                    current_batch_timestamp, # Pass the actual timestamp
                    st.session_state.current_processed_count,
                    current_latest_sentiments
                )
                progress_bar.progress(min(st.session_state.current_processed_count / TOTAL_ROWS, 1.0))
                time.sleep(0.3)
                continue
        
        st.balloons()
        st.success("Analysis complete!")
        st.session_state.is_first_run_complete = True

# --- Display Update Function ---
def update_display(
    keyword_data,                 # Now accepts the full keyword_data dictionary
    chart_placeholder,
    timestamp_display,            # Changed from batch_display
    processed_display,
    current_sentiments_display_placeholder,
    current_timestamp,            # The actual timestamp for the current batch
    processed_count,
    current_latest_sentiments     # Dictionary of latest sentiments for each keyword
):
    
    # Update counters in sidebar
    timestamp_display.markdown(f"`{current_timestamp.strftime('%Y-%m-%d %H:%M:%S')}`")
    processed_display.markdown(f"`{processed_count:,}`")
    
    # Build combined DataFrame for plotting
    all_chart_data = []
    for kw, data in keyword_data.items():
        if not data['timestamps']: # Skip if no data for this keyword yet
            continue
        df_kw = pd.DataFrame({
            'Timestamp': data['timestamps'], # Changed to Timestamp
            'Confidence': data['confidences'],
            'Leading Sentiment': data['labels']
        })
        df_kw['Keyword'] = kw # Add the keyword column
        all_chart_data.append(df_kw)

    if all_chart_data:
        chart_data = pd.concat(all_chart_data, ignore_index=True)
    else: # Fallback if no data at all (shouldn't happen with initial batch logic)
        chart_data = pd.DataFrame(columns=['Timestamp', 'Confidence', 'Leading Sentiment', 'Keyword'])

    # Define colors for the two keywords
    keyword_names = list(keyword_data.keys())
    keyword_colors = {}
    if len(keyword_names) >= 1:
        keyword_colors[keyword_names[0]] = '#1f77b4' # A common blue for the first keyword
    if len(keyword_names) >= 2:
        keyword_colors[keyword_names[1]] = '#ff7f0e'  # A common orange for the second keyword
    # You can add more colors here if you intend to extend beyond 2 keywords

    fig = px.line(
        chart_data,
        x='Timestamp', # Changed to Timestamp
        y='Confidence',
        color='Keyword', # Now color by Keyword
        title="Leading Sentiment Confidence Trend for Keywords' Context",
        labels={'Confidence': 'Confidence Score', 'Timestamp': 'Time of Last Record'}, # Updated label
        height=400,
        hover_data={'Leading Sentiment': True, 'Confidence': ':.1%', 'Keyword': False, 'Timestamp': '|%Y-%m-%d %H:%M:%S'}, # Show sentiment, hide duplicate keyword in hover, format timestamp
        color_discrete_map=keyword_colors if len(keyword_names) == 2 else None # Apply custom colors if exactly 2 keywords
    )
    
    fig.update_layout(
        yaxis_tickformat=".0%",
        hovermode="x unified",
        legend_title_text="Keyword", # Added legend title back
        xaxis_title="Time of Last Record", # Updated axis title
        yaxis_title="Confidence Score",
        # Set y-axis range to 0-1 for sentiment confidence
        yaxis=dict(range=[0, 1])
    )
    
    fig.update_traces(
        mode="markers+lines",
        marker=dict(size=8)
    )
    
    chart_placeholder.plotly_chart(fig, use_container_width=True)

    # Display current sentiments for both keywords
    sentiment_html = "<h4>Current Keyword Sentiments:</h4>"
    cols_sentiment = current_sentiments_display_placeholder.columns(len(current_latest_sentiments))
    
    for idx, (kw, sentiment_info) in enumerate(current_latest_sentiments.items()):
        cols_sentiment[idx].markdown(
            f"<div class='keyword-sentiment-box'><b>{kw}:</b> "
            f"<span class='{sentiment_info['label']}'>{sentiment_info['label'].capitalize()}</span> "
            f"(Confidence: {sentiment_info['confidence']:.1%})</div>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()