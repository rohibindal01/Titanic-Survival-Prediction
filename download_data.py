"""
download_data.py
────────────────
Downloads the Titanic dataset from Kaggle using the Kaggle API.

Prerequisites
─────────────
1. pip install kaggle
2. Create a Kaggle API token:
   → kaggle.com → Account → Create New API Token → downloads kaggle.json
3. Place kaggle.json at:
   - Linux/macOS : ~/.kaggle/kaggle.json
   - Windows     : C:\\Users\\<username>\\.kaggle\\kaggle.json
4. chmod 600 ~/.kaggle/kaggle.json   (Linux/macOS only)
"""

import os
import sys
import zipfile

def download_titanic():
    try:
        import kaggle  # noqa: F401 – triggers credential check
    except ImportError:
        print("❌  kaggle package not found. Run:  pip install kaggle")
        sys.exit(1)

    os.makedirs("data", exist_ok=True)
    print("⬇️  Downloading Titanic dataset from Kaggle...")

    os.system("kaggle competitions download -c titanic -p data")

    # Unzip
    zip_path = "data/titanic.zip"
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall("data")
        os.remove(zip_path)
        print("✅  Data extracted to 'data/'")
        print("    Files: data/train.csv, data/test.csv, data/gender_submission.csv")
    else:
        print("⚠️  Zip not found – check Kaggle credentials or download manually.")
        print("    https://www.kaggle.com/competitions/titanic/data")


if __name__ == "__main__":
    download_titanic()
