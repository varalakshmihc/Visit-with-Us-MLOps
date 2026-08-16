
import os

DATA_PATH = "tourism_project/data/tourism.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

print(f"Dataset registered successfully: {DATA_PATH}")
