"""Streamlit application for simple text emotion recognition."""

import csv
import io
import os
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


EMOTIONS = ("anger", "fear", "joy", "sadness", "surprise")
MODEL_PATH = Path(os.getenv("EMOTION_MODEL_PATH", "emotion_model.pkl"))

EMOTION_EMOJIS = {
    "anger": "😠",
    "fear": "😨",
    "joy": "😊",
    "sadness": "😢",
    "surprise": "😮",
}

# A small baseline classifier with no additional model dependency. Replace
# this dictionary with model inference when a trained model is available.
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
        return pickle.load(model_file)


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


def classify_text(text: str, model: Any | None = None) -> str:
    """Classify text with the trained model or the keyword fallback."""
    if model is None:
        return classify_with_keywords(text)
    if not hasattr(model, "predict"):
        raise TypeError("The loaded model must provide a predict() method.")
    prediction = model.predict([text])
    if len(prediction) != 1:
        raise ValueError("The model must return exactly one prediction per text.")
    return normalize_prediction(prediction[0])


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


def emotion_percentages(emotions: list[str]) -> dict[str, float]:
    """Calculate each emotion's percentage while preserving display order."""
    counts = Counter(emotions)
    total = len(emotions)
    return {emotion: counts[emotion] / total * 100 for emotion in EMOTIONS}


def show_results(texts: list[str], title: str, model: Any | None = None) -> None:
    """Render classification results and aggregate emotion statistics."""
    predictions = [classify_text(text, model) for text in texts]
    percentages = emotion_percentages(predictions)

    st.subheader(title)
    if len(texts) == 1:
        emotion = predictions[0]
        st.success(f"Result: {EMOTION_EMOJIS[emotion]} {emotion.capitalize()}")
    else:
        st.success(f"Result: analyzed {len(texts)} text entries")

    st.subheader("Emotion statistics")
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
                st.write(
                    f"{index}. **{EMOTION_EMOJIS[prediction]} "
                    f"{prediction.capitalize()}** — {text}"
                )


st.set_page_config(page_title="Emotions Detection", page_icon="🔍", layout="wide")
st.title("🔍 Emotions Detection")
st.write(
    "Classify a text or a CSV column into one of five emotions: "
    "anger, fear, joy, sadness, and surprise."
)

try:
    trained_model = load_trained_model(str(MODEL_PATH))
except (AttributeError, EOFError, ImportError, OSError, pickle.UnpicklingError) as error:
    trained_model = None
    st.error(f"Could not load the trained model at `{MODEL_PATH}`: {error}")

if trained_model is None and MODEL_PATH.is_file():
    st.warning("Using the keyword-based fallback classifier instead.")
elif trained_model is None:
    st.info(
        f"No trained model found at `{MODEL_PATH}`. "
        "The app is using the keyword-based fallback classifier."
    )
else:
    st.success(f"Using trained model: `{MODEL_PATH}`")


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
            except (TypeError, ValueError) as error:
                st.error(f"Could not use the trained model: {error}")
    elif text_input.strip():
        try:
            show_results([text_input.strip()], "Text analysis", trained_model)
        except (TypeError, ValueError) as error:
            st.error(f"Could not use the trained model: {error}")
    else:
        st.warning("Upload a CSV file or enter text before defining an emotion.")