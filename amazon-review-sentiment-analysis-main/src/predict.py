"""Predict sentiment with a saved BERT or LSTM model."""

import argparse
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Negative", "Neutral", "Positive"]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.30)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.classifier(self.dropout(hidden[-1]))


def predict_bert(text, device):
    model_dir = ROOT / "models" / "bert_model"
    if not model_dir.exists():
        raise FileNotFoundError("BERT model not found. Train it first using src/train_model.py.")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)[0]
    index = int(probs.argmax().item())
    return LABELS[index], float(probs[index].item())


def predict_lstm(text, device):
    checkpoint_path = ROOT / "models" / "lstm_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError("LSTM model not found. Train it first using src/train_model.py.")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab = checkpoint["vocab"]
    max_length = checkpoint["max_length"]
    tokens = clean_text(text).split()[:max_length]
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens] or [vocab["<UNK>"]]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(ids)], dtype=torch.long)
    model = LSTMClassifier(len(vocab), checkpoint["embedding_dim"], checkpoint["hidden_dim"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(input_ids, lengths), dim=1)[0]
    index = int(probs.argmax().item())
    return LABELS[index], float(probs[index].item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bert", "lstm"], default="bert")
    parser.add_argument("--text", required=True, help="Review text to classify")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label, confidence = predict_bert(args.text, device) if args.model == "bert" else predict_lstm(args.text, device)
    print(f"Sentiment: {label}")
    print(f"Model confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
