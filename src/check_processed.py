from pathlib import Path
import pandas as pd

base_dir = Path(__file__).resolve().parent.parent
processed_path = base_dir / "data" / "processed" / "processed_dataset.csv"

df = pd.read_csv(processed_path)
print(df.head())
print("\nColumns:", df.columns.tolist())
print("\nShape:", df.shape)