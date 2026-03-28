from huggingface_hub import hf_hub_download
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

# Load dataset info to get file list
dataset_info_path = BASE_DIR / "data/dataset/train/dataset_info.json"
save_dir = BASE_DIR / "data/pdfs"

os.makedirs(save_dir, exist_ok=True)

# Read the dataset info
with open(dataset_info_path, "r") as f:
    info = json.load(f)

# Get all PDF file paths from checksums
all_files = list(info["download_checksums"].keys())

# Take only first 10 PDFs
files_to_download = all_files[:]

print(f"Total files available: {len(all_files)}")
print(f"Downloading first 10 PDFs...\n")

for i, hf_path in enumerate(files_to_download):
    # Extract the relative path inside the repo
    # Format: hf://datasets/theatticusproject/cuad@<commit>/CUAD_v1/...
    repo_file_path = hf_path.split("cuad@")[1]           # get part after @
    repo_file_path = repo_file_path.split("/", 1)[1]      # remove commit hash prefix

    # Extract just the filename for saving
    filename = os.path.basename(repo_file_path)
    save_path = os.path.join(save_dir, filename)

    print(f"[{i+1}/10] Downloading: {filename}")
    try:
        downloaded = hf_hub_download(
            repo_id="theatticusproject/cuad",
            repo_type="dataset",
            filename=repo_file_path,
            local_dir=save_dir,
        )
        print(f"       Saved to: {downloaded}")
    except Exception as e:
        print(f"       ERROR: {e}")

print(f"\nDone! PDFs saved to: {save_dir}")