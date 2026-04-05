from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import BertTokenizerFast
from torch.optim import AdamW
from tqdm import tqdm

from src.model import MultiTaskBertForSentiment, MultiTaskSentimentOutput

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "processed_dataset.csv"
TOKENIZER_DIR = BASE_DIR / "models" / "tokenizer"
MODEL_DIR = BASE_DIR / "models" / "bert_multitask"

RANDOM_SEED = 42
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, polarity2id, emotion2id, tone2id, max_len=64):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.polarity2id = polarity2id
        self.emotion2id = emotion2id
        self.tone2id = tone2id
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = str(row["model_text"])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "polarity_labels": torch.tensor(self.polarity2id[row["polarity"]], dtype=torch.long),
            "emotion_labels": torch.tensor(self.emotion2id[row["emotion"]], dtype=torch.long),
            "tone_labels": torch.tensor(self.tone2id[row["tone"]], dtype=torch.long),
            "intensity_labels": torch.tensor(float(row["intensity"]), dtype=torch.float),
        }
        return item

def compute_metrics(y_true, y_pred, label_names, title=""):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_micro = f1_score(y_true, y_pred, average="micro")
    print(f"\n{title}")
    print("Accuracy:", round(acc, 4))
    print("Macro-F1:", round(f1_macro, 4))
    print("Micro-F1:", round(f1_micro, 4))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names))
    return acc, f1_macro, f1_micro

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()

        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            polarity_labels=batch["polarity_labels"],
            emotion_labels=batch["emotion_labels"],
            tone_labels=batch["tone_labels"],
            intensity_labels=batch["intensity_labels"],
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    polarity_true, polarity_pred = [], []
    emotion_true, emotion_pred = [], []
    tone_true, tone_pred = [], []
    intensity_true, intensity_pred = [], []

    total_loss = 0.0

    for batch in tqdm(loader, desc="Validation", leave=False):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            polarity_labels=batch["polarity_labels"],
            emotion_labels=batch["emotion_labels"],
            tone_labels=batch["tone_labels"],
            intensity_labels=batch["intensity_labels"],
        )

        total_loss += outputs.loss.item()

        polarity_true.extend(batch["polarity_labels"].cpu().tolist())
        emotion_true.extend(batch["emotion_labels"].cpu().tolist())
        tone_true.extend(batch["tone_labels"].cpu().tolist())
        intensity_true.extend(batch["intensity_labels"].cpu().tolist())

        polarity_pred.extend(torch.argmax(outputs.polarity_logits, dim=1).cpu().tolist())
        emotion_pred.extend(torch.argmax(outputs.emotion_logits, dim=1).cpu().tolist())
        tone_pred.extend(torch.argmax(outputs.tone_logits, dim=1).cpu().tolist())
        intensity_pred.extend(outputs.intensity_pred.cpu().tolist())

    return {
        "loss": total_loss / len(loader),
        "polarity_true": polarity_true,
        "polarity_pred": polarity_pred,
        "emotion_true": emotion_true,
        "emotion_pred": emotion_pred,
        "tone_true": tone_true,
        "tone_pred": tone_pred,
        "intensity_true": intensity_true,
        "intensity_pred": intensity_pred,
    }

def main():
    set_seed(RANDOM_SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_PATH)

    polarity_labels = sorted(df["polarity"].unique().tolist())
    emotion_labels = sorted(df["emotion"].unique().tolist())
    tone_labels = sorted(df["tone"].unique().tolist())

    polarity2id = {label: i for i, label in enumerate(polarity_labels)}
    emotion2id = {label: i for i, label in enumerate(emotion_labels)}
    tone2id = {label: i for i, label in enumerate(tone_labels)}

    id2polarity = {v: k for k, v in polarity2id.items()}
    id2emotion = {v: k for k, v in emotion2id.items()}
    id2tone = {v: k for k, v in tone2id.items()}

    tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_DIR)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df["polarity"]
    )

    train_dataset = SentimentDataset(train_df, tokenizer, polarity2id, emotion2id, tone2id, MAX_LEN)
    val_dataset = SentimentDataset(val_df, tokenizer, polarity2id, emotion2id, tone2id, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    from transformers import BertConfig
    config = BertConfig.from_pretrained("bert-base-uncased")
    config.vocab_size = len(tokenizer)

    model = MultiTaskBertForSentiment(
        config,
        num_polarity_labels=len(polarity_labels),
        num_emotion_labels=len(emotion_labels),
        num_tone_labels=len(tone_labels),
    )

    model.resize_token_embeddings(len(tokenizer))
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)

    print("Device:", DEVICE)
    print("Train size:", len(train_dataset))
    print("Validation size:", len(val_dataset))
    print("Polarity labels:", polarity_labels)
    print("Emotion labels:", emotion_labels)
    print("Tone labels:", tone_labels)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_results = evaluate(model, val_loader)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_results['loss']:.4f}")

        compute_metrics(
            val_results["polarity_true"],
            val_results["polarity_pred"],
            polarity_labels,
            title="Polarity Metrics"
        )
        compute_metrics(
            val_results["emotion_true"],
            val_results["emotion_pred"],
            emotion_labels,
            title="Emotion Metrics"
        )
        compute_metrics(
            val_results["tone_true"],
            val_results["tone_pred"],
            tone_labels,
            title="Tone Metrics"
        )

        if val_results["loss"] < best_val_loss:
            best_val_loss = val_results["loss"]
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)

            with open(MODEL_DIR / "label_maps.json", "w") as f:
                json.dump(
                    {
                        "polarity2id": polarity2id,
                        "emotion2id": emotion2id,
                        "tone2id": tone2id,
                        "id2polarity": id2polarity,
                        "id2emotion": id2emotion,
                        "id2tone": id2tone,
                    },
                    f,
                    indent=2
                )

            print("Saved best model to:", MODEL_DIR)

    print("\nTraining finished.")

if __name__ == "__main__":
    main()