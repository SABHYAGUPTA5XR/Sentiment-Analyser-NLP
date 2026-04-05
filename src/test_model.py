from pathlib import Path
import pandas as pd
import torch
from transformers import BertConfig, BertTokenizerFast
from model import MultiTaskBertForSentiment

BASE_DIR = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = BASE_DIR / "models" / "tokenizer"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "processed_dataset.csv"

def main():
    df = pd.read_csv(PROCESSED_PATH)
    tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_DIR)

    polarity_labels = sorted(df["polarity"].unique().tolist())
    emotion_labels = sorted(df["emotion"].unique().tolist())
    tone_labels = sorted(df["tone"].unique().tolist())

    num_polarity_labels = len(polarity_labels)
    num_emotion_labels = len(emotion_labels)
    num_tone_labels = len(tone_labels)

    config = BertConfig.from_pretrained("bert-base-uncased")
    config.vocab_size = len(tokenizer)

    model = MultiTaskBertForSentiment(
        config,
        num_polarity_labels=num_polarity_labels,
        num_emotion_labels=num_emotion_labels,
        num_tone_labels=num_tone_labels,
    )

    sample_texts = df["model_text"].head(2).tolist()
    enc = tokenizer(
        sample_texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    outputs = model(**enc)

    print("Tokenizer vocab size:", len(tokenizer))
    print("Number of polarity labels:", num_polarity_labels)
    print("Number of emotion labels:", num_emotion_labels)
    print("Number of tone labels:", num_tone_labels)
    print("\nInput IDs shape:", enc["input_ids"].shape)
    print("Attention mask shape:", enc["attention_mask"].shape)
    print("Polarity logits shape:", outputs.polarity_logits.shape)
    print("Emotion logits shape:", outputs.emotion_logits.shape)
    print("Tone logits shape:", outputs.tone_logits.shape)
    print("Intensity prediction shape:", outputs.intensity_pred.shape)
    print("Token tone logits shape:", outputs.token_tone_logits.shape)
    print("Token emotion logits shape:", outputs.token_emotion_logits.shape)

if __name__ == "__main__":
    main()