from pathlib import Path
from transformers import BertTokenizerFast

BASE_DIR = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = BASE_DIR / "models" / "tokenizer"

def main():
    tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_DIR)

    texts = [
        "<TONE_CASUAL> I can't believe this!!!",
        "<TONE_FORMAL> The service was excellent overall.",
        "<EMO_SAD> I regret buying this product."
    ]

    for text in texts:
        encoded = tokenizer(text)
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        print("\nTEXT:", text)
        print("TOKENS:", tokens)
        print("TOKEN IDS:", encoded["input_ids"])

if __name__ == "__main__":
    main()