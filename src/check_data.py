from pathlib import Path
import pandas as pd

base_dir = Path(__file__).resolve().parent.parent
csv_path = base_dir / "data" / "raw" / "dataset_before_preprocessing.csv"

df = pd.read_csv(csv_path)

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

print("\nPolarity distribution:")
print(df["polarity"].value_counts())

print("\nEmotion distribution:")
print(df["emotion"].value_counts())

print("\nTone distribution:")
print(df["tone"].value_counts())