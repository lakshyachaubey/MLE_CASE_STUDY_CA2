"""Simple Streamlit interface for the fine-tuned BERT classifier."""

from pathlib import Path
import torch
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "bert_model"
LABELS = ["Negative", "Neutral", "Positive"]

st.set_page_config(page_title="Amazon Review Sentiment Analyzer", page_icon="📊")
st.title("Amazon Review Sentiment Analyzer")
st.write("Enter a product review to classify it as negative, neutral, or positive.")


@st.cache_resource
def load_model():
    if not MODEL_DIR.exists():
        raise FileNotFoundError("BERT model not found. Train the model first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
    return tokenizer, model, device


review = st.text_area("Product review", height=150, placeholder="The product works perfectly and I am very happy with the purchase.")
if st.button("Analyze sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        try:
            tokenizer, model, device = load_model()
            inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                probabilities = torch.softmax(model(**inputs).logits, dim=1)[0].cpu()
            index = int(probabilities.argmax().item())
            st.subheader(f"Predicted sentiment: {LABELS[index]}")
            st.metric("Model confidence", f"{probabilities[index].item():.1%}")
            st.bar_chart({LABELS[i]: float(probabilities[i]) for i in range(3)})
        except FileNotFoundError as exc:
            st.error(str(exc))
