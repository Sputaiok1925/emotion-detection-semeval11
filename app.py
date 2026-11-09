"""Streamlit application for simple text emotion recognition."""

import csv
import io
import os
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


EMOTIONS = ("anger", "fear", "joy", "sadness", "surprise")
MODEL_OPTIONS = {
    "BERT CLS pooling": "logreg_cls.joblib",
    "BERT mean pooling": "logreg_mean.joblib",
}
BERT_MODEL_NAME = "bert-base-uncased"
BERT_MODEL_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
BERT_MAX_LENGTH = 128

EMOTION_EMOJIS = {
    "anger": "😠",
    "fear": "😨",
    "joy": "😊",
    "sadness": "😢",
    "surprise": "😮",
}

# Dependency-free fallback used when no trained model is available.
EMOTION_KEYWORDS = {
    "anger": {
        "angry",
        "anger",
        "annoyed",
        "furious",
        "hate",
        "hated",
        "irritated",
        "mad",
        "rage",
        "rude",
        "upset",
    },
    "fear": {
        "afraid",
        "anxious",
        "danger",
        "fear",
        "frightened",
        "horror",
        "nervous",
        "panic",
        "scared",
        "terrified",
        "worry",
        "worried",
    },
    "joy": {
        "amazing",
        "enjoy",
        "excited",
        "fun",
        "glad",
        "good",
        "happy",
        "joy",
        "laugh",
        "love",
        "loved",
        "smile",
        "wonderful",
    },
    "sadness": {
        "alone",
        "cry",
        "depressed",
        "disappointed",
        "grief",
        "lonely",
        "miss",
        "sad",
        "sorry",
        "tear",
        "unhappy",
    },
    "surprise": {
        "astonished",
        "unexpected",
        "surprise",
        "surprised",
        "shocked",
        "suddenly",
        "unbelievable",
        "wow",
    },
}


@st.cache_resource
def load_trained_model(model_path: str) -> Any | None:
    """Load a serialized model if one has been provided."""
    path = Path(model_path)
    if not path.is_file():
        return None
    with path.open("rb") as model_file:
        if path.suffix.lower() == ".joblib":
            return joblib.load(model_file)
        return pickle.load(model_file)


@st.cache_resource
def load_bert_encoder() -> tuple[Any, Any, Any]:
    """Load the tokenizer and encoder used to create the model features."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        BERT_MODEL_NAME,
        revision=BERT_MODEL_REVISION,
    )
    encoder = AutoModel.from_pretrained(
        BERT_MODEL_NAME,
        revision=BERT_MODEL_REVISION,
    )
    encoder.eval()
    return tokenizer, encoder, torch


def classify_with_bert_model(
    texts: list[str], model: dict[str, Any]
) -> list[list[str]]:
    """Apply the saved per-emotion classifiers to BERT text embeddings."""
    model_emotions = model.get("emotions")
    classifiers = model.get("models")
    thresholds = model.get("thresholds")
    metadata = model.get("meta")
    pooling = metadata.get("pooling") if isinstance(metadata, dict) else None
    if (
        model_emotions != list(EMOTIONS)
        or not isinstance(classifiers, list)
        or len(classifiers) != len(EMOTIONS)
        or not isinstance(thresholds, list)
        or len(thresholds) != len(EMOTIONS)
        or pooling not in {"cls", "mean"}
        or "scaler" not in model
    ):
        raise ValueError("The saved BERT model has an unsupported format.")

    tokenizer, encoder, torch = load_bert_encoder()
    tokens = tokenizer(
        texts,
        max_length=BERT_MAX_LENGTH,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        token_embeddings = encoder(**tokens).last_hidden_state

    if pooling == "cls":
        embeddings = token_embeddings[:, 0, :]
    else:
        attention_mask = tokens["attention_mask"].unsqueeze(-1)
        masked_embeddings = token_embeddings * attention_mask
        embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(
            dim=1
        ).clamp(min=1)

    scaled_embeddings = model["scaler"].transform(embeddings.cpu().numpy())
    emotion_probabilities = np.column_stack(
        [
            classifier.predict_proba(scaled_embeddings)[
                :, list(classifier.classes_).index(1)
            ]
            for classifier in classifiers
        ]
    )

    predictions: list[list[str]] = []
    for probabilities in emotion_probabilities:
        positive_emotions = [
            index
            for index, (probability, threshold) in enumerate(
                zip(probabilities, thresholds)
            )
            if probability >= threshold
        ]
        if not positive_emotions:
            positive_emotions = [int(np.argmax(probabilities))]
        predictions.append([EMOTIONS[index] for index in positive_emotions])
    return predictions


def classify_with_keywords(text: str) -> str:
    """Classify text with the dependency-free fallback classifier."""
    words = set(re.findall(r"[A-Za-z]+", text.lower()))
    scores = {
        emotion: len(words.intersection(keywords))
        for emotion, keywords in EMOTION_KEYWORDS.items()
    }

    highest_score = max(scores.values())
    if highest_score == 0:
        return "joy"
    return max(EMOTIONS, key=lambda emotion: scores[emotion])


def normalize_prediction(prediction: Any) -> str:
    """Validate a model prediction against the emotions supported by the UI."""
    emotion = str(prediction).strip().lower()
    if emotion not in EMOTIONS:
        raise ValueError(
            f"Model returned {prediction!r}; expected one of {', '.join(EMOTIONS)}."
        )
    return emotion


def classify_text(text: str, model: Any | None = None) -> list[str]:
    """Classify text with the trained model or the keyword fallback."""
    return classify_texts([text], model)[0]


def classify_texts(texts: list[str], model: Any | None = None) -> list[list[str]]:
    """Classify multiple texts in one model call when possible."""
    if model is None:
        return [[classify_with_keywords(text)] for text in texts]
    if isinstance(model, dict):
        return classify_with_bert_model(texts, model)
    if not hasattr(model, "predict"):
        raise TypeError("The loaded model must provide a predict() method.")
    predictions = model.predict(texts)
    if len(predictions) != len(texts):
        raise ValueError("The model must return one prediction per text.")
    normalized_predictions = []
    for prediction in predictions:
        labels = [prediction] if isinstance(prediction, str) else prediction
        normalized_labels = list(dict.fromkeys(
            normalize_prediction(label) for label in labels
        ))
        if not normalized_labels:
            raise ValueError("The model must return at least one emotion per text.")
        normalized_predictions.append(normalized_labels)
    return normalized_predictions


def read_csv_texts(uploaded_file: io.BytesIO) -> list[str]:
    """Read text values from an uploaded CSV file.

    A column named ``text`` is preferred. If it is not present, the first
    non-empty column is used so that simple one-column CSV files also work.
    """
    raw_data = uploaded_file.getvalue()
    try:
        decoded_data = raw_data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("The CSV must be encoded as UTF-8.") from error

    rows = list(csv.reader(io.StringIO(decoded_data)))
    if not rows:
        raise ValueError("The uploaded CSV file is empty.")

    header = [cell.strip().lower() for cell in rows[0]]
    text_column = header.index("text") if "text" in header else None
    start_row = 1 if text_column is not None else 0

    if text_column is None:
        non_empty_columns = [
            index
            for index in range(len(rows[0]))
            if any(row[index].strip() for row in rows if len(row) > index)
        ]
        if not non_empty_columns:
            raise ValueError("The uploaded CSV does not contain any text.")
        text_column = non_empty_columns[0]

    texts = [
        row[text_column].strip()
        for row in rows[start_row:]
        if len(row) > text_column and row[text_column].strip()
    ]
    if not texts:
        raise ValueError("The uploaded CSV does not contain any non-empty text rows.")
    return texts


def emotion_percentages(predictions: list[list[str]]) -> dict[str, float]:
    """Calculate each emotion's percentage while preserving display order."""
    counts = Counter(emotion for row in predictions for emotion in set(row))
    total = len(predictions)
    return {emotion: counts[emotion] / total * 100 for emotion in EMOTIONS}


def show_results(texts: list[str], title: str, model: Any | None = None) -> None:
    """Render classification results and aggregate emotion statistics."""
    predictions = classify_texts(texts, model)
    percentages = emotion_percentages(predictions)

    st.subheader(title)
    if len(texts) == 1:
        emotions = predictions[0]
        result = ", ".join(
            f"{EMOTION_EMOJIS[emotion]} {emotion.capitalize()}"
            for emotion in emotions
        )
        st.success(f"Result: {result}")
    else:
        st.success(f"Result: analyzed {len(texts)} text entries")

    st.subheader("Emotion statistics (% of texts; totals may exceed 100%)")
    chart_data = pd.DataFrame(
        {
            "Emotion": list(percentages),
            "Percentage": list(percentages.values()),
        }
    )
    st.bar_chart(chart_data, x="Emotion", y="Percentage")

    columns = st.columns(len(EMOTIONS))
    for column, emotion in zip(columns, EMOTIONS):
        column.metric(
            f"{EMOTION_EMOJIS[emotion]} {emotion.capitalize()}",
            f"{percentages[emotion]:.1f}%",
        )

    if len(texts) > 1:
        with st.expander("View individual predictions"):
            for index, (text, prediction) in enumerate(zip(texts, predictions), 1):
                result = ", ".join(
                    f"{EMOTION_EMOJIS[emotion]} {emotion.capitalize()}"
                    for emotion in prediction
                )
                st.write(
                    f"{index}. **{result}** — {text}"
                )


st.set_page_config(page_title="Emotions Detection", page_icon="🔍", layout="wide")
st.title("🔍 Emotions Detection")
st.write(
    "Classify a text or a CSV column into one of five emotions: "
    "anger, fear, joy, sadness, and surprise."
)

model_path_env = os.getenv("EMOTION_MODEL_PATH")
if model_path_env:
    model_path = Path(model_path_env)
else:
    selected_model = st.selectbox("Model", list(MODEL_OPTIONS))
    model_path = (
        Path(__file__).resolve().parent / "models" / MODEL_OPTIONS[selected_model]
    )

try:
    trained_model = load_trained_model(str(model_path))
except (
    AttributeError,
    EOFError,
    ImportError,
    OSError,
    pickle.UnpicklingError,
    ValueError,
) as error:
    trained_model = None
    st.error(f"Could not load the trained model at `{model_path}`: {error}")

if trained_model is None and model_path.is_file():
    st.warning("Using the keyword-based fallback classifier instead.")
elif trained_model is None:
    st.info(
        f"No trained model found at `{model_path}`. "
        "The app is using the keyword-based fallback classifier."
    )
else:
    st.success(f"Using trained model: `{model_path}`")


tab1, tab2 = st.tabs(["📁 Upload CSV", "✍️ Write text"])

with tab1:
    uploaded_file = st.file_uploader(
        "Load a CSV file",
        type=["csv"]
    )

with tab2:
    text_input = st.text_area(
        "Write your text here",
        placeholder="Example: I am so happy to see you!",
        height=140,
    )

if st.button("Define emotion", type="primary"):
    if uploaded_file is not None:
        try:
            csv_texts = read_csv_texts(uploaded_file)
        except ValueError as error:
            st.error(str(error))
        else:
            try:
                show_results(csv_texts, "CSV analysis", trained_model)
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
                st.error(f"Could not use the trained model: {error}")
    elif text_input.strip():
        try:
            show_results([text_input.strip()], "Text analysis", trained_model)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            st.error(f"Could not use the trained model: {error}")
    else:
        st.warning("Upload a CSV file or enter text before defining an emotion.")