"""Train LSTM and BERT models on Amazon product reviews.

Expected CSV columns: reviewText and overall (rating 1-5).
Run from the repository root:
    python src/train_model.py --data data/amazon_reviews.csv
"""

from __future__ import annotations

import argparse
import json
import re
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
LABEL_NAMES = ["Negative", "Neutral", "Positive"]
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def rating_to_label(rating: float) -> int:
    if rating <= 2:
        return 0
    if rating == 3:
        return 1
    return 2


def load_and_split(data_path: Path, max_samples: int, seed: int):
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Download the Amazon reviews CSV and place it at data/amazon_reviews.csv."
        )

    df = pd.read_csv(data_path, on_bad_lines="skip")
    required = {"reviewText", "overall"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required column(s): {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[["reviewText", "overall"]].copy()
    df["overall"] = pd.to_numeric(df["overall"], errors="coerce")
    df = df.dropna(subset=["reviewText", "overall"])
    df["reviewText"] = df["reviewText"].astype(str).str.strip()
    df = df[(df["reviewText"] != "") & df["overall"].isin([1, 2, 3, 4, 5])]
    df = df.drop_duplicates(subset=["reviewText", "overall"])
    df["label"] = df["overall"].apply(rating_to_label).astype(int)

    if df["label"].nunique() != 3:
        raise ValueError(
            "The cleaned dataset must contain all three sentiment classes. "
            "Check the rating distribution."
        )

    if max_samples > 0 and len(df) > max_samples:
        df, _ = train_test_split(
            df,
            train_size=max_samples,
            random_state=seed,
            stratify=df["label"],
        )

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["label"],
    )
    return df.reset_index(drop=True), train_df, val_df, test_df


def build_vocab(texts: List[str], max_vocab: int) -> Dict[str, int]:
    counts: Counter = Counter()
    for text in texts:
        counts.update(clean_text(text).split())
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, _ in counts.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def encode_lstm(text: str, vocab: Dict[str, int], max_length: int) -> List[int]:
    ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in clean_text(text).split()[:max_length]]
    return ids or [vocab[UNK_TOKEN]]


class ReviewDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], max_length: int):
        self.sequences = [encode_lstm(text, vocab, max_length) for text in texts]
        self.labels = [int(label) for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.tensor(self.sequences[index], dtype=torch.long), torch.tensor(self.labels[index], dtype=torch.long)


def lstm_collate(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded, lengths, torch.stack(labels)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.30)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.classifier(self.dropout(hidden[-1]))


def calculate_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def save_confusion_matrix(y_true, y_pred, output_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def train_lstm(train_df, val_df, test_df, args, device, output_dir: Path):
    train_texts = train_df["reviewText"].tolist()
    val_texts = val_df["reviewText"].tolist()
    test_texts = test_df["reviewText"].tolist()
    vocab = build_vocab(train_texts, args.max_vocab)

    train_ds = ReviewDataset(train_texts, train_df["label"].tolist(), vocab, args.max_length)
    val_ds = ReviewDataset(val_texts, val_df["label"].tolist(), vocab, args.max_length)
    test_ds = ReviewDataset(test_texts, test_df["label"].tolist(), vocab, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.lstm_batch_size, shuffle=True, collate_fn=lstm_collate)
    val_loader = DataLoader(val_ds, batch_size=args.lstm_batch_size, shuffle=False, collate_fn=lstm_collate)
    test_loader = DataLoader(test_ds, batch_size=args.lstm_batch_size, shuffle=False, collate_fn=lstm_collate)

    model = LSTMClassifier(len(vocab)).to(device)
    counts = np.bincount(train_df["label"].to_numpy(), minlength=3)
    weights = len(train_df) / (3.0 * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lstm_learning_rate)

    start = time.perf_counter()
    for epoch in range(args.lstm_epochs):
        model.train()
        losses = []
        for sequences, lengths, labels in tqdm(train_loader, desc=f"LSTM epoch {epoch + 1}"):
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(sequences, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        val_true, val_pred = [], []
        with torch.no_grad():
            for sequences, lengths, labels in val_loader:
                logits = model(sequences.to(device), lengths)
                val_pred.extend(logits.argmax(dim=1).cpu().tolist())
                val_true.extend(labels.tolist())
        print(f"LSTM epoch {epoch + 1}: loss={np.mean(losses):.4f}, validation_accuracy={accuracy_score(val_true, val_pred):.4f}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - start

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for sequences, lengths, labels in test_loader:
            logits = model(sequences.to(device), lengths)
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.tolist())

    metrics = calculate_metrics(y_true, y_pred)
    metrics["training_seconds"] = training_seconds
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "max_length": args.max_length,
            "embedding_dim": 128,
            "hidden_dim": 128,
            "label_names": LABEL_NAMES,
        },
        ROOT / "models" / "lstm_model.pt",
    )
    save_confusion_matrix(y_true, y_pred, output_dir / "confusion_matrix_lstm.png", "LSTM confusion matrix")
    report = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=LABEL_NAMES, zero_division=0)
    (output_dir / "classification_report_lstm.txt").write_text(report, encoding="utf-8")
    return metrics


def train_bert(train_df, val_df, test_df, args, device, output_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.bert_model, num_labels=3).to(device)

    class TextDataset(Dataset):
        def __init__(self, frame):
            self.texts = frame["reviewText"].astype(str).tolist()
            self.labels = frame["label"].astype(int).tolist()
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, index):
            encoded = tokenizer(
                self.texts[index], truncation=True, padding="max_length",
                max_length=args.bert_max_length, return_tensors="pt"
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[index], dtype=torch.long),
            }

    train_loader = DataLoader(TextDataset(train_df), batch_size=args.bert_batch_size, shuffle=True)
    val_loader = DataLoader(TextDataset(val_df), batch_size=args.bert_batch_size, shuffle=False)
    test_loader = DataLoader(TextDataset(test_df), batch_size=args.bert_batch_size, shuffle=False)

    counts = np.bincount(train_df["label"].to_numpy(), minlength=3)
    weights = len(train_df) / (3.0 * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.bert_learning_rate)

    start = time.perf_counter()
    for epoch in range(args.bert_epochs):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"BERT epoch {epoch + 1}"):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = criterion(output.logits, batch["labels"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        val_true, val_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                val_pred.extend(logits.argmax(dim=1).cpu().tolist())
                val_true.extend(batch["labels"].cpu().tolist())
        print(f"BERT epoch {epoch + 1}: loss={np.mean(losses):.4f}, validation_accuracy={accuracy_score(val_true, val_pred):.4f}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - start

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(batch["labels"].cpu().tolist())

    metrics = calculate_metrics(y_true, y_pred)
    metrics["training_seconds"] = training_seconds
    model_dir = ROOT / "models" / "bert_model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    save_confusion_matrix(y_true, y_pred, output_dir / "confusion_matrix_bert.png", "BERT confusion matrix")
    report = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=LABEL_NAMES, zero_division=0)
    (output_dir / "classification_report_bert.txt").write_text(report, encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train LSTM and BERT on Amazon product reviews.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "amazon_reviews.csv")
    parser.add_argument("--max-samples", type=int, default=50000, help="Maximum rows to use; use 0 for all rows.")
    parser.add_argument("--max-vocab", type=int, default=30000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lstm-epochs", type=int, default=5)
    parser.add_argument("--bert-epochs", type=int, default=2)
    parser.add_argument("--lstm-batch-size", type=int, default=64)
    parser.add_argument("--bert-batch-size", type=int, default=8)
    parser.add_argument("--lstm-learning-rate", type=float, default=1e-3)
    parser.add_argument("--bert-learning-rate", type=float, default=2e-5)
    parser.add_argument("--bert-model", type=str, default="bert-base-uncased")
    parser.add_argument("--skip-bert", action="store_true", help="Train only the LSTM baseline.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    (ROOT / "models").mkdir(exist_ok=True)
    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df, train_df, val_df, test_df = load_and_split(args.data, args.max_samples, args.seed)
    print(f"Rows after cleaning: {len(df):,}")
    print("Sentiment counts:")
    print(df["label"].map(dict(enumerate(LABEL_NAMES))).value_counts())

    # Save dataset summaries and EDA figures.
    counts = df["label"].value_counts().reindex([0, 1, 2], fill_value=0)
    plt.figure(figsize=(7, 5))
    sns.barplot(x=[LABEL_NAMES[i] for i in counts.index], y=counts.values)
    plt.title("Amazon review sentiment distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of reviews")
    plt.tight_layout()
    plt.savefig(output_dir / "sentiment_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="overall", color="steelblue")
    plt.title("Amazon product rating distribution")
    plt.xlabel("Star rating")
    plt.ylabel("Number of reviews")
    plt.tight_layout()
    plt.savefig(output_dir / "rating_distribution.png", dpi=160)
    plt.close()

    df["review_length"] = df["reviewText"].astype(str).str.split().str.len()
    plt.figure(figsize=(8, 5))
    sns.histplot(df["review_length"].clip(upper=500), bins=50)
    plt.title("Review length distribution (capped at 500 words)")
    plt.xlabel("Word count")
    plt.tight_layout()
    plt.savefig(output_dir / "review_length_distribution.png", dpi=160)
    plt.close()

    all_metrics = {}
    all_metrics["LSTM"] = train_lstm(train_df, val_df, test_df, args, device, output_dir)
    if not args.skip_bert:
        all_metrics["BERT"] = train_bert(train_df, val_df, test_df, args, device, output_dir)

    (output_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    comparison = pd.DataFrame(all_metrics).T
    comparison.to_csv(output_dir / "model_comparison.csv", index_label="model")
    comparison[["accuracy", "macro_precision", "macro_recall", "macro_f1"]].plot(kind="bar", figsize=(9, 5))
    plt.title("LSTM vs BERT: test-set performance")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.close()
    print("\nFinished. Metrics saved to results/metrics.json")
    print(comparison.round(4))


if __name__ == "__main__":
    main()
