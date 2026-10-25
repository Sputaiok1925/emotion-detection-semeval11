# Bridging the Gap in Text-Based Emotion Detection

This project is part of the [NLP course](https://peter.bednar.website.tuke.sk/nlp/index.html) at TUKE. It is a team project divided into two stages. The first stage involves designing the project and developing a baseline solution. The second stage focuses on implementing our own solution and evaluating its performance.

Git Repository: https://github.com/Sputaiok1925/emotion-detection-semeval11

## Project Overview

**Problem Statement**

Detecting emotions in text is challenging because the same words can express different emotions depending on the context. Simple approaches may fail to capture these nuances, while more complex deep learning models require significant computational resources.

**Motivation**

Text is one of the most common forms of communication, allowing people to express and share their thoughts. However, identifying the emotions behind a text message can be difficult, especially when emojis or other explicit emotional cues are absent. Our motivation is to explore NLP techniques for emotion detection and find a balance between classification performance, model complexity, and computational cost. We also aim to gain practical experience in implementing and evaluating an NLP system.

**Project Goal**

Our goal is to develop a system that detects emotions in English text using a pretrained BERT model with a frozen encoder and a separate classifier. We aim to evaluate its performance and determine whether this approach provides a reasonable balance between accuracy and computational efficiency.

## Datasets

We use the [BRIGHTER dataset](https://brighter-dataset.github.io/) for our assignment. It is a multilingual, human-annotated dataset for text-based emotion detection, covering 28 languages and various text domains. It includes six emotion categories: anger, disgust, fear, joy, sadness, and surprise. The dataset supports multi-label classification, allowing a text to be associated with multiple emotions.

## Existing methods

- **Lexicon-based:** Use predefined word–emotion dictionaries (e.g., NRC Emotion Lexicon or WordNet-Affect) to estimate emotions from the presence of words associated with particular emotional categories. These methods require no model training and have low computational costs. The NRC Emotion Lexicon, for example, was developed using crowdsourced word–emotion annotations [6]. However, lexicon-based methods have limited ability to interpret context, negation, irony, and other complex linguistic patterns, which can reduce their effectiveness on informal or ambiguous text [7]. Their performance depends on the quality and coverage of the lexicon, so it is not appropriate to assign one general accuracy range to all lexicon-based systems.

- **Classical ML (TF-IDF + SVM/Logistic Regression):** Convert text into numerical features using bag-of-words or TF-IDF, then train a classifier such as Logistic Regression or a Support Vector Machine. These methods are relatively easy to implement, computationally efficient, and useful as baseline models for text classification [7, 9]. They can learn associations between words and emotion labels, but standard bag-of-words representations do not explicitly capture word order or contextual meaning [9]. Sebastian Raschka identifies bag-of-words combined with Logistic Regression as a practical baseline for text classification [9]. In one study using the GoEmotions dataset, TF-IDF + SVM achieved 72.1% accuracy, compared with 85.2% for the study's fine-tuned BERT model [10]. These figures are specific to that experiment and should not be interpreted as typical accuracy values for these methods in every emotion-detection task.

- **Deep Learning (CNN/RNN):** These models use dense word embeddings and neural networks to learn task-specific patterns in text. *CNNs* apply convolutional filters over word representations to identify local patterns. Kim (2014) demonstrated that a relatively simple CNN achieved state-of-the-art results on four of seven sentence-classification benchmarks, including sentiment analysis and question classification [11]. *RNNs/LSTMs* process text sequentially and use recurrent connections to represent dependencies between tokens; the original LSTM architecture was introduced by Hochreiter and Schmidhuber [13]. These approaches can learn richer patterns than simple bag-of-words models, but their performance depends on the dataset, embeddings, architecture, and training configuration [7, 11]. For example, Raschka reports approximately 90.07% accuracy for a CNN and 85.66% for an LSTM in illustrative experiments on the IMDb sentiment-classification dataset [9]. These are sentiment-analysis results, not emotion-detection results, and therefore should not be used as direct evidence of expected performance on our task. Neural models also require more implementation and training effort than a basic TF-IDF classifier, although the computational requirements vary considerably with model size.

- **Transformer-based:** Transformer models use self-attention to process relationships between tokens and support parallel sequence processing [3]. BERT builds contextual representations by considering both left and right context, and its pretrained parameters can be adapted to a downstream task through fine-tuning [12]. RoBERTa further demonstrated that pretraining choices, training data, and optimization can substantially affect the performance of pretrained language models [4].

  - **Fine-tuned BERT:** Add a classification head and train it together with the pretrained BERT encoder. This allows the model's representations to adapt to the target task, but training requires backpropagation through the encoder and therefore more computational resources than training a separate classifier on fixed embeddings [8, 12]. In one GoEmotions experiment, the fine-tuned BERT model achieved 85.2% accuracy, outperforming the TF-IDF + SVM and BiLSTM baselines reported in the same study [10]. This is evidence that fine-tuned BERT can perform well in a particular experimental setup, not proof that it always achieves the highest accuracy.

  - **Frozen BERT + classifier:** Use pretrained BERT as a feature extractor while keeping its parameters fixed. The output token representations are converted into a single fixed-length text vector, for example by selecting the `[CLS]` representation or applying mean or max pooling. A separate classifier, such as Logistic Regression or SVM, is then trained on the resulting vectors [8, 16]. If the embeddings are precomputed and cached, the encoder does not need to be run again during classifier training, which can reduce training time and resource requirements [8, 16]. Research on frozen BERT representations has shown that this approach can be competitive with fine-tuning on certain text-classification tasks, although results depend on the task and the pooling architecture [16]. The main limitation is that BERT itself cannot adapt its representations to the emotion labels. Therefore, frozen BERT does not guarantee the same or better accuracy than fine-tuning; this should be evaluated experimentally.

  - **Adapters/LoRA:** Parameter-efficient methods retain the pretrained model weights and train a smaller number of additional parameters. Adapter modules add trainable components to the network, while LoRA introduces trainable low-rank matrices into selected layers. The original adapter study achieved results within 0.8 percentage points of full fine-tuning on GLUE while adding 3.6% additional parameters per task [14]. LoRA similarly aims to reduce the number of trainable parameters and GPU memory requirements [15]. These methods offer alternatives when full fine-tuning is too expensive, but they introduce additional implementation choices compared with using a frozen encoder and a separate classifier.

  - **Sentence-BERT (SBERT):** SBERT modifies pretrained BERT or RoBERTa using siamese or triplet network structures to produce sentence embeddings that are more suitable for semantic similarity comparisons [5]. The paper evaluates `[CLS]`, mean pooling, and max pooling, with mean pooling used as its default configuration [5]. It also reports reducing the time for finding similar sentence pairs among 10,000 sentences from approximately 65 hours with BERT/RoBERTa to about five seconds with SBERT [5]. This result concerns semantic similarity search, not emotion-classification accuracy. SBERT is also distinct from simply freezing an unmodified BERT encoder and training a separate emotion classifier.

- **Multimodal approaches (briefly):** Some emotion-detection systems combine text with additional modalities, such as speech or facial information. These approaches introduce the need to collect and process additional signals, and their results depend on the available modalities and experimental setup [7]. Since our project focuses on text-based emotion detection, multimodal approaches are outside the current scope.

## Baseline Solution

**Method Selection**

We use a pretrained BERT model with a frozen encoder and a separate classification layer. This approach allows us to benefit from BERT's ability to capture contextual information without the computational cost of fine-tuning the entire model.

**Model Architecture**

1. **Tokenization.** The input text is converted into a sequence of tokens using the BERT tokenizer, including the required special tokens.

2. **Contextual Representation.** The pretrained BERT encoder generates contextual representations for each token. Its weights remain frozen during training.

3. **Text Representation.** The token representations are averaged to obtain a single vector representing the entire text. Padding tokens are excluded from the average, and the handling of special tokens is kept consistent throughout the experiments.

4. **Emotion Classification.** Five independent binary logistic regression classifiers are trained on the resulting text vectors. Each classifier predicts the presence or absence of one emotion, allowing the system to assign multiple emotions to the same text.

5. **Thresholding.** The predicted probabilities are converted into binary labels using a decision threshold. A threshold of 0.5 serves as the initial baseline, while alternative thresholds may be selected using the development set (`dev`).

**Data Splitting and Evaluation**

The dataset's official split for the selected version is preserved to ensure a consistent and fair evaluation.

- **Train:** Used to train the logistic regression classifiers.
- **Dev:** Used to select model settings and classification thresholds.
- **Test:** Used only for the final evaluation after all model decisions and thresholds have been fixed.

## Evaluation Methodology

The following metrics will be used to evaluate the performance of the English text emotion detection system:

| Metric | Description |
|---|---|
| **Macro-F1** | The primary evaluation metric. It calculates the average F1-score across all emotion categories, giving each category equal weight regardless of its frequency in the dataset. |
| **Precision** | Measures the proportion of predicted emotions that are correct. |
| **Recall** | Measures the proportion of actual emotions that the model correctly identifies. |
| **F1-score** | The harmonic mean of Precision and Recall, used to evaluate the performance for each emotion category. |

## Using LLM and AI agents

Different LLMs and AI agents were used in this project, including ChatGPT for research and documentation editing, and GitHub Copilot for software implementation.

## Sources & References

[1] Kashfia Sailunaz, Manmeet Dhaliwal, Jon Rokne, and Reda Alhajj. Emotion detection from text and speech: a survey. Social Network Analysis and Mining, 8:1–26, 2018.

[2] Erdenebileg Batbaatar, Meijing Li, and Keun Ho Ryu. Semanticemotion neural network for emotion recognition from text. IEEE Access, 7:111866–111878, 201.
   
[3] Vaswani, A., et al. (2017). *Attention Is All You Need.* Advances in Neural Information Processing Systems, 30. https://papers.nips.cc/paper/7181-attention-is-all-you-need

[4] Liu, Y., et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv:1907.11692. https://arxiv.org/abs/1907.11692

[5] Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* Proceedings of EMNLP-IJCNLP 2019, 3982–3992. https://doi.org/10.18653/v1/D19-1410

[6] Mohammad, S. M., & Turney, P. D. (2013). *Crowdsourcing a Word–Emotion Association Lexicon.* Computational Intelligence, 29(3), 436–465. https://doi.org/10.1111/j.1467-8640.2012.00460.x

[7] Kusal, S., Patil, S., Choudrie, J., Kotecha, K., Vora, D., & Pappas, I. (2022). *A Review on Text-Based Emotion Detection—Techniques, Applications, Datasets, and Future Directions.* arXiv:2205.03235. https://arxiv.org/abs/2205.03235

[8] Singh, S. K. (2025, August 2). *Using BERT: Strategies for Text Classification.* Medium. https://medium.com/@shubham.ksingh.cer14/using-bert-strategies-for-text-classification-32f9c211576d

[9] Raschka, S. (2026, September 29). *Language Models for Text Classification: From Bag-of-Words to Jev.* Sebastian Raschka. https://magazine.sebastianraschka.com/p/classifier-history-and-jev

[10] Sagar, P., Ruban, S., & Vimalrajan, M. (2025). *Improving Fine-Grained Emotion Detection in Text with BERT and GoEmotions.* Premier Journal of Science, 15, 100204. Online publication: January 27, 2026. https://doi.org/10.70389/PJS.100204

[11] Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification.* Proceedings of EMNLP 2014, 1746–1751. https://doi.org/10.3115/v1/D14-1181

[12] Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* Proceedings of NAACL-HLT 2019, 4171–4186. https://doi.org/10.18653/v1/N19-1423

[13] Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

[14] Houlsby, N., et al. (2019). *Parameter-Efficient Transfer Learning for NLP.* Proceedings of the 36th International Conference on Machine Learning, 2790–2799. https://proceedings.mlr.press/v97/houlsby19a.html

[15] Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685. https://arxiv.org/abs/2106.09685

[16] Galal, O., Abdel-Gawad, A. H., & Farouk, M. (2024). *Rethinking of BERT Sentence Embedding for Text Classification.* Neural Computing and Applications, 36, 20245–20258. https://doi.org/10.1007/s00521-024-10212-3

[17] Demszky, D., et al. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions.* Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 4040–4054. https://doi.org/10.18653/v1/2020.acl-main.372
