import kagglehub
import json
import os

# Load Kaggle credentials
with open('kaggle.json', 'r') as f:
    credentials = json.load(f)

# Set environment variables for Kaggle authentication
os.environ['KAGGLE_USERNAME'] = credentials['username']
os.environ['KAGGLE_KEY'] = credentials['key']

# Download latest version
path = kagglehub.dataset_download("tobiasgrass/unet-data-png")

print("Path to dataset files:", path)