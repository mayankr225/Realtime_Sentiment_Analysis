import pickle
import numpy as np

# Load the saved model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def classify_sentiment(text):
    # Vectorize the text
    X = vectorizer.transform([text])

    # Make prediction
    prediction = model.predict(X)[0]

    # Get decision function scores for confidence approximation
    decision_scores = model.decision_function(X)[0]

    # Apply softmax to convert to pseudo-probabilities
    exp_scores = np.exp(decision_scores)
    confidence_scores = exp_scores / exp_scores.sum()

    # Map prediction to sentiment label and confidence score
    if prediction == 1:
        sentiment = "Positive"
        confidence = confidence_scores[2]
    elif prediction == -1:
        sentiment = "Negative"
        confidence = confidence_scores[0]
    else:
        sentiment = "Neutral"
        confidence = confidence_scores[1]

    return sentiment, confidence

def main():
    text = input("Enter a text: ")
    sentiment, confidence = classify_sentiment(text)
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence * 100:.2f}%")

if __name__ == "__main__":
    main()
