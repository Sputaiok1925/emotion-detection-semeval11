# emotion-detection-semeval11
Bridging the Gap in Text-Based Emotion Detection.

## Streamlit application

The project includes a Streamlit interface for classifying text into:

- anger
- fear
- joy
- sadness
- surprise

Run it with:

```bash
pip install -r requirements.txt
streamlit run app.py
```

You can enter one text directly or upload a UTF-8 CSV file. For CSV files,
the app uses a column named `text` when available; otherwise it uses the first
non-empty column. The results include the predicted emotion and the percentage
of each emotion across all analyzed rows.

The app includes two logistic-regression models that use BERT embeddings.
`BERT CLS pooling` is selected by default; `BERT mean pooling` is also
available in the model selector. On the first prediction, the app downloads
the pinned `bert-base-uncased` encoder used to create the model features.
Internet access is required for that initial download.
Each text can have multiple predicted emotions when multiple model scores
meet their thresholds. CSV emotion percentages show the share of texts assigned
each emotion, so they can add up to more than 100%.

You can use a different serialized model with:

```bash
EMOTION_MODEL_PATH=models/my_model.joblib streamlit run app.py
```

The custom model can be a `.joblib` file or a pickle file that exposes
`predict(list_of_texts)` and returns labels matching the five supported emotion
names. If no model file is available, the app uses the keyword-based fallback
classifier. Only load model files you trust, because serialized Python model
files can execute code while being loaded.
