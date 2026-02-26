import kagglehub
import pandas as pd
from pathlib import Path

print("Downloading dataset from Kaggle...")
dataset_path = kagglehub.dataset_download("paperxd/all-computer-prices")
print(f"Dataset downloaded to: {dataset_path}")

df = kagglehub.dataset_load(
    kagglehub.KaggleDatasetAdapter.PANDAS,
    "paperxd/all-computer-prices",
    "computer_prices_all.csv"
)

output_path = Path(__file__).parent / "data" / "raw" / "computer_prices_all.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Data saved to: {output_path}")
print(f"Shape: {df.shape}")
