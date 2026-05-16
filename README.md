# Toxic Comment Classification with DistilBERT

A fine-tuned NLP pipeline that classifies toxic online comments using DistilBERT, built in Google Colab as a personal deep learning project. The goal was to work through every stage of a real transformer-based pipeline from data loading and tokenisation to training, evaluation, and inference.

---

## Overview

This project fine-tunes a multilingual DistilBERT model on a subset of the [Jigsaw Toxic Comment dataset](https://www.kaggle.com/code/tanulsingh077/deep-learning-for-nlp-zero-to-transformers-bert/notebook) to detect whether a comment is toxic or not. It covers the full NLP pipeline, including preprocessing, model building, handling class imbalance, and evaluating on both multilingual and English-only test sets.

Google Colab Notebook: https://drive.google.com/file/d/1CqCUVrRnqad9yw31ipC-frzfAuC-6kxF/view?usp=sharing

---

## Motivation

I wanted to move beyond standard classification tasks and get hands-on experience with transformer models. Fine-tuning a pre-trained model like DistilBERT on a real-world, messy dataset felt like the right way to build genuine intuition for how modern NLP works under the hood.

The Jigsaw dataset also presented a practical challenge: class imbalance. Toxic comments are a minority in real data, so I had to think carefully about how to train and evaluate the model without it simply learning to predict "non-toxic" for everything.

---

## Features

- End-to-end NLP pipeline built and run entirely in Google Colab
- Fine-tuned DistilBERT with a binary classification head
- Weighted BCE loss to handle class imbalance during training
- Tokenisation using HuggingFace's `distilbert-base-multilingual-cased`
- Evaluation on both a multilingual test set and an English-only subset
- Performance visualised with confusion matrices, ROC-AUC curves, and Plotly charts
- Saved model weights for reuse and inference

---

## How It Works

1. **Data loading:** A subset of the Jigsaw training data is loaded along with separate validation and test splits.
2. **Preprocessing:** Comments are tokenised and encoded using the DistilBERT multilingual tokenizer, with padding and attention masks applied.
3. **Model:** A pre-trained DistilBERT backbone is used with a single linear classification layer on top of the CLS token output.
4. **Training:** The model is trained with Adam optimiser and weighted BCEWithLogitsLoss, which penalises missed toxic comments more heavily to offset the class imbalance.
5. **Evaluation:** Accuracy and ROC-AUC are computed on both test sets to measure how well the model generalises across languages.

```python
class ToxicClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-multilingual-cased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits
```

---

## Results

| Metric   | Multilingual Test | English-only Test |
|----------|:-----------------:|:-----------------:|
| Accuracy | 81.88%            | 85.70%            |
| ROC-AUC  | 87.56%            | 94.27%            |

The gap between multilingual and English-only performance reflects the inherent difficulty of cross-lingual generalisation, a known challenge with multilingual models.

---

## Why DistilBERT

DistilBERT retains around 97% of BERT's performance at roughly 40% fewer parameters. For a Colab environment with limited GPU memory and runtime, that trade-off made practical sense. Using the multilingual variant also meant the model could handle comments written in languages other than English without any additional language-specific preprocessing.

---

## Tech Stack

- Python 3
- HuggingFace Transformers (`distilbert-base-multilingual-cased`)
- PyTorch
- Scikit-learn
- Matplotlib, Seaborn, Plotly
- Google Colab

---

## Project Structure

```
Toxic-Comment-Classification-with-DistilBERT/
│
├── jigsaw-toxic-comment-train.csv   # Training dataset
├── validation.csv                   # Validation dataset
├── test.csv                         # Multilingual test data
├── test_labels.csv                  # Ground-truth test labels
├── tokenizer/                       # Tokenizer config files
├── toxic_model_v1.pt                # Saved model weights
├── NLP Toxic Comment New.ipynb      # Main Colab notebook
└── README.md
```

---

## How to Run

```bash
# Step 1: Clone the repository inside Google Colab
!git clone https://github.com/2801/Toxic-Comment-Classification-with-DistilBERT.git
%cd Toxic-Comment-Classification-with-DistilBERT

# Step 2: Check GPU availability
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Step 3: Load data
import pandas as pd
train = pd.read_csv('jigsaw-toxic-comment-train.csv', nrows=10000)
valid = pd.read_csv('validation.csv')
test  = pd.read_csv('test.csv', nrows=5000)

# Step 4: Load tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')
```

Then open and run `NLP Toxic Comment New.ipynb` from top to bottom.

---

## Learning Outcomes

- Understood how fine-tuning a pre-trained transformer differs from training a model from scratch, and why it is so effective
- Learned how the CLS token in BERT-style models captures sentence-level representations for classification
- Saw firsthand how class imbalance affects training, and how weighted loss functions address it without resampling
- Gained practical experience with HuggingFace's Transformers library, including tokenizers, attention masks, and model configs
- Understood the performance gap between multilingual and monolingual evaluation, and what it reveals about cross-lingual transfer

---

## Future Improvements

- Add attention heatmaps to visualise which tokens drive toxic predictions
- Extend to multi-label classification to detect subtypes of toxicity
- Experiment with full BERT or RoBERTa for a performance comparison
- Apply threshold tuning to improve recall on the toxic minority class
