import re
import string
import pickle
import numpy as np
import pandas as pd
import streamlit as st

import torch
import torch.nn as nn


# =========================
# Model Architecture
# =========================

class HybridEmbeddingMLP(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        max_len,
        feature_size,
        hidden1,
        hidden2,
        num_classes,
        dropout_rate
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        embedding_size = max_len * embedding_dim
        input_size = embedding_size + feature_size

        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x_seq, x_features):
        embedded = self.embedding(x_seq)
        embedded = embedded.view(embedded.shape[0], -1)

        combined = torch.cat((embedded, x_features), dim=1)

        output = self.classifier(combined)

        return output


# =========================
# Text Preprocessing
# =========================

def basic_clean_text(text):
    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

    text = re.sub(r"\d+", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def handle_negation(text):
    words = str(text).split()

    new_words = []

    i = 0

    while i < len(words):
        if words[i] in {"not", "no", "never"} and i + 1 < len(words):
            new_words.append(words[i] + "_" + words[i + 1])
            i += 2
        else:
            new_words.append(words[i])
            i += 1

    return " ".join(new_words)


positive_words = {
    "good", "great", "excellent", "amazing", "perfect",
    "love", "best", "nice", "awesome", "fantastic"
}

negative_words = {
    "bad", "worst", "terrible", "broken", "poor",
    "hate", "awful", "disappointed", "defective", "waste"
}


def extract_extra_features(text):
    text = str(text).lower()
    words = text.split()

    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)

    exclamation_count = text.count("!")
    question_count = text.count("?")

    review_length = len(words)

    return [
        positive_count,
        negative_count,
        exclamation_count,
        question_count,
        review_length
    ]


# =========================
# Embedding Input
# =========================

def tokenize(text):
    return str(text).lower().split()


def review_to_indices(review, word_to_ix, max_len):
    words = tokenize(review)

    indices = []

    for word in words:
        if word in word_to_ix:
            indices.append(word_to_ix[word])
        else:
            indices.append(word_to_ix["<UNK>"])

    if len(indices) < max_len:
        indices = indices + [word_to_ix["<PAD>"]] * (max_len - len(indices))
    else:
        indices = indices[:max_len]

    return indices


# =========================
# Load Files
# =========================

@st.cache_resource
def load_artifacts():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("word_to_ix.pkl", "rb") as f:
        word_to_ix = pickle.load(f)

    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    with open("tfidf_ngram_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("extra_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("ex3_config.pkl", "rb") as f:
        config = pickle.load(f)

    model = HybridEmbeddingMLP(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        max_len=config["max_len"],
        feature_size=config["feature_size"],
        hidden1=config["hidden1"],
        hidden2=config["hidden2"],
        num_classes=config["num_classes"],
        dropout_rate=config["dropout_rate"]
    )

    model.load_state_dict(
        torch.load(
            "ex3_sentiment_model.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model, word_to_ix, encoder, vectorizer, scaler, config, device


# =========================
# Prediction
# =========================

def predict_single_review(review, model, word_to_ix, encoder, vectorizer, scaler, config, device):
    model.eval()

    clean_review = basic_clean_text(review)

    advanced_review = handle_negation(clean_review)

    review_seq = np.array([
        review_to_indices(
            review=clean_review,
            word_to_ix=word_to_ix,
            max_len=config["max_len"]
        )
    ])

    review_tfidf = vectorizer.transform(
        [advanced_review]
    ).toarray().astype("float32")

    extra_features = np.array([
        extract_extra_features(clean_review)
    ])

    extra_features_scaled = scaler.transform(extra_features).astype("float32")

    final_features = np.hstack(
        [review_tfidf, extra_features_scaled]
    ).astype("float32")

    review_seq_tensor = torch.LongTensor(review_seq).to(device)

    features_tensor = torch.FloatTensor(final_features).to(device)

    with torch.no_grad():
        outputs = model(review_seq_tensor, features_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_class = torch.max(probabilities, dim=1)

    predicted_sentiment = encoder.inverse_transform(
        [predicted_class.item()]
    )[0]

    class_names = list(encoder.classes_)

    probability_dict = {}

    for index, class_name in enumerate(class_names):
        probability_dict[class_name] = probabilities[0][index].item()

    return {
        "review": review,
        "clean_review": clean_review,
        "advanced_review": advanced_review,
        "predicted_sentiment": predicted_sentiment,
        "confidence": confidence.item(),
        "probabilities": probability_dict
    }


def predict_batch_reviews(reviews, model, word_to_ix, encoder, vectorizer, scaler, config, device):
    results = []

    for review in reviews:
        prediction = predict_single_review(
            review=review,
            model=model,
            word_to_ix=word_to_ix,
            encoder=encoder,
            vectorizer=vectorizer,
            scaler=scaler,
            config=config,
            device=device
        )

        row = {
            "review": prediction["review"],
            "predicted_sentiment": prediction["predicted_sentiment"],
            "confidence": prediction["confidence"]
        }

        for class_name, probability in prediction["probabilities"].items():
            row[f"{class_name.lower()}_probability"] = probability

        results.append(row)

    return pd.DataFrame(results)


def sentiment_distribution_summary(results_df):
    total_reviews = len(results_df)

    counts = results_df["predicted_sentiment"].value_counts()

    positive_count = counts.get("Positive", 0)
    neutral_count = counts.get("Neutral", 0)
    negative_count = counts.get("Negative", 0)

    positive_percentage = (positive_count / total_reviews) * 100 if total_reviews > 0 else 0
    neutral_percentage = (neutral_count / total_reviews) * 100 if total_reviews > 0 else 0
    negative_percentage = (negative_count / total_reviews) * 100 if total_reviews > 0 else 0

    percentages = {
        "Positive": positive_percentage,
        "Neutral": neutral_percentage,
        "Negative": negative_percentage
    }

    overall_mood = max(percentages, key=percentages.get)

    summary = {
        "total_reviews": total_reviews,
        "positive_count": positive_count,
        "positive_percentage": positive_percentage,
        "neutral_count": neutral_count,
        "neutral_percentage": neutral_percentage,
        "negative_count": negative_count,
        "negative_percentage": negative_percentage,
        "overall_dataset_mood": overall_mood
    }

    return summary


# =========================
# Streamlit App
# =========================

st.set_page_config(
    page_title="Sentiment Analysis System",
    layout="wide"
)

st.title("Sentiment Analysis System")

st.write(
    "Experiment 3: Embedding + TF-IDF N-grams + Negation Handling + Extra Features + MLP"
)

model, word_to_ix, encoder, vectorizer, scaler, config, device = load_artifacts()

tab1, tab2 = st.tabs(
    [
        "Single Review Prediction",
        "Batch Dataset Prediction"
    ]
)


# =========================
# Single Review
# =========================

with tab1:
    st.header("Single Review Prediction")

    review = st.text_area(
        "Enter a review",
        height=150,
        placeholder="Example: The product is not good and delivery was late"
    )

    if st.button("Predict Sentiment"):
        if review.strip() == "":
            st.warning("Please enter a review first.")
        else:
            prediction = predict_single_review(
                review=review,
                model=model,
                word_to_ix=word_to_ix,
                encoder=encoder,
                vectorizer=vectorizer,
                scaler=scaler,
                config=config,
                device=device
            )

            st.subheader("Prediction Result")

            st.write("Predicted Sentiment:", prediction["predicted_sentiment"])

            st.write("Confidence:", round(prediction["confidence"] * 100, 2), "%")

            probabilities_df = pd.DataFrame({
                "sentiment": list(prediction["probabilities"].keys()),
                "probability": [
                    value * 100 for value in prediction["probabilities"].values()
                ]
            })

            st.dataframe(probabilities_df)

            st.bar_chart(
                probabilities_df.set_index("sentiment")
            )

            with st.expander("Preprocessing Details"):
                st.write("Clean Review:", prediction["clean_review"])
                st.write("Advanced Review:", prediction["advanced_review"])


# =========================
# Batch Reviews
# =========================

with tab2:
    st.header("Batch Dataset Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV file containing reviews",
        type=["csv"]
    )

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.dataframe(batch_df.head())

        text_column = st.selectbox(
            "Select the review text column",
            batch_df.columns
        )

        if st.button("Run Batch Prediction"):
            reviews = batch_df[text_column].astype(str).tolist()

            results_df = predict_batch_reviews(
                reviews=reviews,
                model=model,
                word_to_ix=word_to_ix,
                encoder=encoder,
                vectorizer=vectorizer,
                scaler=scaler,
                config=config,
                device=device
            )

            summary = sentiment_distribution_summary(results_df)

            st.subheader("Prediction Results Preview")
            st.dataframe(results_df.head())

            st.subheader("Sentiment Distribution Summary")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total Reviews",
                summary["total_reviews"]
            )

            col2.metric(
                "Positive",
                f'{summary["positive_count"]} ({summary["positive_percentage"]:.2f}%)'
            )

            col3.metric(
                "Neutral",
                f'{summary["neutral_count"]} ({summary["neutral_percentage"]:.2f}%)'
            )

            col4.metric(
                "Negative",
                f'{summary["negative_count"]} ({summary["negative_percentage"]:.2f}%)'
            )

            st.success(
                f'Overall Dataset Mood: {summary["overall_dataset_mood"]}'
            )

            chart_df = pd.DataFrame({
                "sentiment": ["Positive", "Neutral", "Negative"],
                "count": [
                    summary["positive_count"],
                    summary["neutral_count"],
                    summary["negative_count"]
                ]
            })

            st.bar_chart(
                chart_df.set_index("sentiment")
            )

            output_csv = results_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction Results CSV",
                data=output_csv,
                file_name="sentiment_predictions.csv",
                mime="text/csv"
            )