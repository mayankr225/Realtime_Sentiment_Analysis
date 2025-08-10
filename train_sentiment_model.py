import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
import pickle

nltk.download('vader_lexicon')

def generate_labels(data):
    sia = SentimentIntensityAnalyzer()
    sentiments = []
    for text in data['cleaned_text']:
        if not isinstance(text, str):
            text = ""
        sentiment_scores = sia.polarity_scores(text)
        if sentiment_scores['compound'] >= 0.05:
            sentiment = 1
        elif sentiment_scores['compound'] <= -0.05:
            sentiment = -1
        else:
            sentiment = 0
        sentiments.append(sentiment)
    data['sentiment'] = sentiments
    return data

def train_model(data):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=10000
    )
    X = vectorizer.fit_transform(data['cleaned_text'])
    y = data['sentiment']

    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    param_grid = {
        'alpha': [0.0001, 0.001, 0.01],
        'penalty': ['l2', 'l1'],
        'loss': ['log_loss', 'hinge']
    }

    clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
    grid = GridSearchCV(clf, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
    grid.fit(X_resampled, y_resampled)

    best_model = grid.best_estimator_

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Best parameters found:", grid.best_params_)

    return best_model, vectorizer, accuracy, report

def main():
    data = pd.read_csv('cleaned_comments.csv')

    # REMOVE rows with missing 'cleaned_text'
    data = data.dropna(subset=['cleaned_text']).reset_index(drop=True)

    data = generate_labels(data)
    model, vectorizer, accuracy, report = train_model(data)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print("Model trained and saved successfully.")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)

if __name__ == "__main__":
    main()
