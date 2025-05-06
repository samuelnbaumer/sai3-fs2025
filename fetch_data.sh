#!/bin/bash

# Set target directory
TARGET_DIR="datasets/"
mkdir -p "$TARGET_DIR"

# Base URL for dataset files (in 'data/' folder)
BASE_URL="https://huggingface.co/datasets/corbt/all-recipes/resolve/main/data"

# List of dataset files to download
FILES=(
  "train-00000-of-00004-237b1b1141fdcfa1.parquet"
  "train-00001-of-00004-d46654ac93566129.parquet"
  "train-00002-of-00004-3b4f78b99eedadc2.parquet"
  "train-00003-of-00004-2369b90eb0860a76.parquet"
)

echo "Downloading All-Recipes dataset into $TARGET_DIR"

# Loop over files and download each
for file in "${FILES[@]}"; do
  echo "Downloading $file..."
  wget -q --show-progress "$BASE_URL/$file" -O "$TARGET_DIR/$file"
done

echo "✅ Done!"
