import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

file_path = BASE_DIR.parent.parent / "data" / "raw" / "computer_prices_all.csv"

df = pd.read_csv(file_path)

print(df.head())