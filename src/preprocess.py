from pathlib import Path
import pandas as pd
import re

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "dataset_before_preprocessing.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "processed_dataset.csv"

CASUAL_WORDS = {
    "lol", "bro", "kinda", "honestly", "btw", "haha", "damn", "dude",
    "okay", "ok", "yup", "nah", "gonna", "wanna", "nice", "cool"
}

FORMAL_WORDS = {
    "therefore", "accordingly", "sincerely", "regarding", "furthermore",
    "however", "hence", "overall", "in summary", "in my opinion", "respectfully"
}

POSITIVE_WORDS = {
    "love", "great", "good", "excellent", "wonderful", "impressed", "fantastic", "happy"
}

NEGATIVE_WORDS = {
    "hate", "terrible", "awful", "bad", "disappointed", "anxious", "stressed",
    "frustrating", "regret"
}

EMOJI_MAP = {
    "😡": "<EMO_ANGRY>",
    "😢": "<EMO_SAD>",
    "😄": "<EMO_HAPPY>",
    "😊": "<EMO_HAPPY>",
    "😐": "<EMO_NEUTRAL>",
    "😞": "<EMO_SAD>",
    "😤": "<EMO_ANGER>",
    "😂": "<EMO_LAUGH>",
}

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\u200b", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_special_tokens(text):
    raw = str(text)
    lower = raw.lower()
    tokens = []

    if raw != raw.lower():
        tokens.append("<CAPS>")

    if re.search(r"\b[A-Z]{3,}\b", raw):
        tokens.append("<ALL_CAPS>")

    exclam = raw.count("!")
    if exclam >= 1:
        tokens.append(f"<EXCLAM_{min(exclam, 3)}>")

    ques = raw.count("?")
    if ques >= 1:
        tokens.append(f"<QUESTION_{min(ques, 3)}>")

    if re.search(r"(.)\1{2,}", lower):
        tokens.append("<ELONGATED>")

    for emo, token in EMOJI_MAP.items():
        if emo in raw:
            tokens.append(token)

    if any(word in lower for word in CASUAL_WORDS):
        tokens.append("<TONE_CASUAL>")

    if any(word in lower for word in FORMAL_WORDS):
        tokens.append("<TONE_FORMAL>")

    if any(word in lower for word in POSITIVE_WORDS):
        tokens.append("<SENT_POS_HINT>")

    if any(word in lower for word in NEGATIVE_WORDS):
        tokens.append("<SENT_NEG_HINT>")

    if not tokens:
        tokens.append("<NEUTRAL_STYLE>")

    return " ".join(tokens)

def main():
    df = pd.read_csv(RAW_PATH)

    rows = []
    for _, row in df.iterrows():
        text = str(row["text"])
        clean = clean_text(text)
        special = extract_special_tokens(text)
        model_text = f"{special} {clean}"

        rows.append({
            "text": text,
            "clean_text": clean,
            "special_tokens": special,
            "model_text": model_text,
            "polarity": row["polarity"],
            "emotion": row["emotion"],
            "tone": row["tone"],
            "intensity": row["intensity"]
        })

    processed_df = pd.DataFrame(rows)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(PROCESSED_PATH, index=False)

    print("Raw shape:", df.shape)
    print("Processed shape:", processed_df.shape)
    print("Saved to:", PROCESSED_PATH)
    print("\nPreview:")
    print(processed_df.head())

if __name__ == "__main__":
    main()