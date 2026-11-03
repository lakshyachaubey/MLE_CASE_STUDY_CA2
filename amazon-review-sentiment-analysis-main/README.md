# Amazon Product Review Sentiment Analysis Using BERT and LSTM

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![NLP](https://img.shields.io/badge/Domain-NLP-orange)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-PyTorch-red)
![Hugging Face](https://img.shields.io/badge/Transformers-Hugging%20Face-yellow)
![Status](https://img.shields.io/badge/Status-Completed%20after%20training-success)

## Overview

This project focuses on performing sentiment analysis on Amazon product reviews using Natural Language Processing (NLP) and deep learning techniques.

The primary objective is to automatically classify customer reviews into three sentiment categories:

- Positive
- Neutral
- Negative

Two deep learning architectures are implemented and compared:

1. **BERT (Bidirectional Encoder Representations from Transformers):** A pretrained transformer model fine-tuned using Hugging Face Transformers.
2. **LSTM (Long Short-Term Memory):** A recurrent neural network developed as a baseline sentiment classification model.

The project compares both models based on classification performance and computational efficiency.

## Problem Statement

E-commerce platforms receive large volumes of customer reviews every day. Manually analyzing these reviews to understand customer satisfaction, product quality, and customer opinions is time-consuming.

This project develops an automated sentiment analysis system that processes Amazon product reviews and predicts their sentiment using deep learning.

## Objectives

- Perform exploratory data analysis on Amazon product reviews.
- Clean and preprocess textual review data.
- Generate sentiment labels using product ratings.
- Implement an LSTM-based sentiment classification model.
- Fine-tune a pretrained BERT transformer using Hugging Face.
- Evaluate both models using standard classification metrics.
- Compare model performance and training efficiency.
- Discuss transfer learning and real-world deployment constraints.

## Dataset

**Dataset:** Amazon Product Reviews

**Format:** CSV

**Primary input feature:** `reviewText`

**Target feature:** `overall`

The dataset contains customer reviews and product ratings.

### Important Dataset Columns

| Column | Description |
|---|---|
| asin | Amazon product identifier |
| helpful | Helpful review votes |
| overall | Product rating from 1 to 5 |
| reviewText | Actual customer review |
| reviewTime | Review date |
| reviewerID | Reviewer identifier |
| reviewerName | Reviewer name |
| summary | Review headline |
| unixReviewTime | Review timestamp |

### Dataset Source

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/yasserh/amazon-product-reviews-dataset

Place the downloaded CSV file inside the `dataset/` directory.

The raw dataset is not included in this repository. Refer to the original dataset provider for its usage and licensing terms.

## Sentiment Label Generation

The dataset contains product ratings rather than manually annotated sentiment labels.

Therefore, ratings are converted into three sentiment classes.

| Product Rating | Sentiment | Encoded Label |
|---|---|---|
| 1–2 Stars | Negative | 0 |
| 3 Stars | Neutral | 1 |
| 4–5 Stars | Positive | 2 |

These labels are rating-based approximations of sentiment and may contain some labeling noise.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Data splitting and evaluation |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | Pretrained BERT implementation |
| NLTK / Regular Expressions | Text preprocessing |
| Streamlit | Interactive application |

## Project Architecture

### 1. Data Preprocessing

- Load Amazon product reviews.
- Handle missing values.
- Remove duplicate reviews.
- Clean review text.
- Generate sentiment labels from ratings.
- Split data into training, validation, and testing sets.

### 2. Exploratory Data Analysis

- Product rating distribution.
- Sentiment class distribution.
- Review length distribution.
- Sentiment percentage analysis.
- Class imbalance analysis.

### 3. LSTM Model

The LSTM model uses:

- Text tokenization.
- Vocabulary generation.
- Word embedding layer.
- LSTM recurrent layer.
- Dropout.
- Fully connected classification layer.

### 4. BERT Model

The BERT implementation uses the pretrained `bert-base-uncased` model.

The model is fine-tuned using Amazon product reviews for three-class sentiment classification.

Key components:

- BERT tokenizer.
- Input IDs.
- Attention masks.
- Pretrained transformer encoder.
- Classification head.
- Fine-tuning using AdamW optimization.

### 5. Model Evaluation

Both models are evaluated using:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion matrix
- Training time
- Inference latency

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/amazon-review-sentiment-analysis.git
```

Navigate to the project directory:

```bash
cd amazon-review-sentiment-analysis
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset and place it inside the `dataset/` directory.

## Execution

Open the Jupyter Notebook:

```bash
jupyter notebook
```

Navigate to:

```text
notebooks/sentiment_analysis.ipynb
```

Execute the notebook cells in order to:

1. Load and preprocess the dataset.
2. Perform exploratory data analysis.
3. Generate sentiment labels.
4. Train the LSTM model.
5. Fine-tune the BERT model.
6. Evaluate both models.
7. Generate performance comparison graphs.

For Google Colab, upload the dataset and run the notebook with GPU acceleration enabled when available.

## Experimental Results

The project compares the LSTM and BERT models using the same training, validation, and testing splits.

| Evaluation Metric | LSTM | BERT |
|---|---|---|
| Accuracy | To be measured | To be measured |
| Macro Precision | To be measured | To be measured |
| Macro Recall | To be measured | To be measured |
| Macro F1-score | To be measured | To be measured |
| Training Time | To be measured | To be measured |
| Inference Latency | To be measured | To be measured |

Actual experimental values should be added after training and evaluation.

## Transfer Learning Analysis

BERT uses pretrained language representations learned from large text corpora.

Fine-tuning allows the pretrained model to adapt these representations to the Amazon review sentiment classification task.

In contrast, the baseline LSTM model in this project learns its word embeddings and sequential patterns from the available training dataset.

This experiment investigates the trade-off between pretrained language understanding, model performance, training requirements, and computational efficiency.

## Real-World Applications

- Customer feedback analysis.
- Product quality monitoring.
- E-commerce customer satisfaction analysis.
- Automated review categorization.
- Brand reputation monitoring.
- Business intelligence and decision-making.

## Deployment Considerations

The project also examines practical challenges associated with deploying NLP models:

- Computational resource requirements.
- Model inference latency.
- Dataset imbalance.
- Data privacy and anonymization.
- Model drift.
- Scalability.
- Sarcasm and ambiguous customer reviews.

Potential optimizations include DistilBERT, model quantization, batch inference, and cloud-based deployment.

## Limitations

- Sentiment labels are generated from product ratings and may not always represent the actual textual sentiment.
- Three-star reviews can contain mixed opinions.
- Sarcasm and contextual ambiguity can affect prediction quality.
- BERT requires more computational resources than the baseline LSTM.
- Model performance depends on dataset quality and class distribution.

## Future Improvements

- Experiment with DistilBERT and RoBERTa.
- Incorporate review summaries as additional input.
- Improve class imbalance handling.
- Explore hyperparameter optimization.
- Deploy the model using Streamlit.
- Implement model quantization for efficient inference.
- Evaluate sentiment trends across different product categories.

---

If you find this project useful, consider giving the repository a star!
