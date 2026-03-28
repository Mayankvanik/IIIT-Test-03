
from datasets import load_dataset, DownloadMode
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

# Define the save path
save_path = BASE_DIR / "data/dataset"

# Create the directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

# Download the dataset (skip split size verification)
print("Downloading CUAD dataset...")
ds = load_dataset(
    "theatticusproject/cuad",
    verification_mode="no_checks"   # ← fixes the NonMatchingSplitsSizesError
)

# Save the dataset to disk
print(f"Saving dataset to {save_path}...")
ds.save_to_disk(save_path)

print("Done! Dataset saved successfully.")
print(f"Dataset splits: {list(ds.keys())}")
for split in ds:
    print(f"  {split}: {len(ds[split])} examples")










    