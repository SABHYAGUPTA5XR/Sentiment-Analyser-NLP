import pandas as pd
import numpy as np
import re
import torch
from datasets import load_dataset

# Mocking emoji availability for the script
EMOJI_AVAILABLE = False 

# ─────────────────────────────────────────────
# 1. CONFIG & MAPPINGS
# ─────────────────────────────────────────────
CONFIG = {
    "max_samples": 2000,
    "polarity_labels": ["negative", "neutral", "positive"],
    "emotion_labels":  ["sadness", "anxiety", "happiness", "anger", "surprise", "neutral"],
    "tone_labels":     ["casual", "formal", "professional"],
}

# ─────────────────────────────────────────────
# 2. PREPROCESSING & PROSODY
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)       # Remove HTML
    text = re.sub(r'http\S+|www\S+', ' ', text) # Remove URLs
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def inject_prosody_tokens(text: str) -> str:
    # Consecutive caps words
    text = re.sub(r'\b[A-Z]{2,}\b', lambda m: "[CAPS] " + m.group(), text)
    # Exclamation marks
    text = re.sub(r'!+', ' [EXCLAIM] ', text)
    # Ellipsis
    text = re.sub(r'\.{2,}', ' [ELLIPSIS] ', text)
    # Question marks
    text = re.sub(r'\?+', ' [QUESTION] ', text)
    return re.sub(r'\s+', ' ', text).strip()

# ─────────────────────────────────────────────
# 3. AUTO-LABELING HEURISTICS (Your Logic)
# ─────────────────────────────────────────────
def label_polarity(text: str, imdb_label: int) -> str:
    weak_positive = ["okay", "ok", "fine", "decent", "average", "not bad", "alright"]
    weak_negative = ["not great", "not good", "disappointing", "mediocre", "so-so"]
    text_lower = text.lower()
    for phrase in weak_positive + weak_negative:
        if phrase in text_lower:
            return "neutral"
    return "positive" if imdb_label == 1 else "negative"

def label_emotion(text: str) -> str:
    text_lower = text.lower()
    rules = {
        "sadness": ["sad", "cry", "grief", "heartbreak", "depressing"],
        "anxiety": ["anxious", "nervous", "worried", "fear", "dread"],
        "happiness": ["happy", "joy", "love", "wonderful", "amazing"],
        "anger": ["angry", "furious", "rage", "hate", "awful"],
        "surprise": ["surprise", "shocked", "unexpected", "wow"],
    }
    scores = {label: 0 for label in rules}
    for label, keywords in rules.items():
        for kw in keywords:
            if kw in text_lower: scores[label] += 1
    
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "neutral"

def label_tone(text: str) -> str:
    text_lower = text.lower()
    casual_cues = ["lol", "omg", "gonna", "hey", "!!!", "haha"]
    formal_cues = ["furthermore", "nevertheless", "thus", "accordingly"]
    prof_cues   = ["recommend", "analysis", "effective", "review"]

    scores = {
        "casual": sum(1 for c in casual_cues if c in text_lower),
        "formal": sum(1 for c in formal_cues if c in text_lower),
        "professional": sum(1 for c in prof_cues if c in text_lower)
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "casual"

def label_intensity(text: str, polarity: str) -> float:
    exclaim_score = min(text.count("!") * 0.15, 0.4)
    caps_ratio    = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    caps_score    = min(caps_ratio * 2, 0.3)
    base          = 0.3 if polarity == "neutral" else 0.5
    return round(min(base + exclaim_score + caps_score, 1.0), 2)

# ─────────────────────────────────────────────
# 4. MAIN EXECUTION
# ─────────────────────────────────────────────
def generate_imdb_styled_dataset():
    print("Loading IMDB dataset...")
    raw = load_dataset("imdb", trust_remote_code=True)
    train_data = raw["train"].select(range(CONFIG["max_samples"]))
    
    dataset_rows = []

    print("Synthesizing labels...")
    for item in train_data:
        original_text = item['text']
        imdb_label = item['label']
        
        # 1. Preprocess
        processed_text = clean_text(original_text)
        # Note: We use original text for labeling to keep cues, 
        # but inject tokens into the final output text.
        final_text = inject_prosody_tokens(processed_text)
        
        # 2. Apply Heuristics
        pol = label_polarity(original_text, imdb_label)
        emo = label_emotion(original_text)
        ton = label_tone(original_text)
        intense = label_intensity(original_text, pol)
        
        dataset_rows.append({
            "text": final_text,
            "polarity": pol,
            "emotion": emo,
            "tone": ton,
            "intensity": intense
        })

    # Save to CSV
    df = pd.DataFrame(dataset_rows)
    df.to_csv("imdb_augmented_dataset.csv", index=False)
    print(f"Success! Saved {len(df)} rows to 'imdb_augmented_dataset.csv'")

if __name__ == "__main__":
    generate_imdb_styled_dataset()