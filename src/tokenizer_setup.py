from pathlib import Path
import pandas as pd
from transformers import BertTokenizerFast

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "processed_dataset.csv"
TOKENIZER_DIR = BASE_DIR / "models" / "tokenizer"

def collect_special_tokens(df):
    tokens = set()

    fixed_tokens = [
        "<CAPS>",
        "<ALL_CAPS>",
        "<ELONGATED>",
        "<TONE_CASUAL>",
        "<TONE_FORMAL>",
        "<SENT_POS_HINT>",
        "<SENT_NEG_HINT>",
        "<NEUTRAL_STYLE>",
        "<EMO_ANGRY>",
        "<EMO_SAD>",
        "<EMO_HAPPY>",
        "<EMO_NEUTRAL>",
        "<EMO_ANGER>",
        "<EMO_LAUGH>",
    ]

    for token in fixed_tokens:
        tokens.add(token)

    for i in range(1, 4):
        tokens.add(f"<EXCLAM_{i}>")
        tokens.add(f"<QUESTION_{i}>")

    if "special_tokens" in df.columns:
        for value in df["special_tokens"].astype(str):
            for tok in value.split():
                if tok.startswith("<") and tok.endswith(">"):
                    tokens.add(tok)

    return sorted(tokens)

def main():
    df = pd.read_csv(PROCESSED_PATH)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    special_tokens = collect_special_tokens(df)

    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(TOKENIZER_DIR)

    print("Processed dataset shape:", df.shape)
    print("Number of special tokens added:", len(special_tokens))
    print("Tokenizer saved to:", TOKENIZER_DIR)
    print("Original vocab size:", tokenizer.vocab_size)
    print("Total vocab size after adding tokens:", len(tokenizer))

    sample_text = "<TONE_CASUAL> I am so happy about this!!!"
    encoded = tokenizer(sample_text)

    print("\nSample text:")
    print(sample_text)
    print("\nTokenized output:")
    print(tokenizer.convert_ids_to_tokens(encoded["input_ids"]))

if __name__ == "__main__":
    main()