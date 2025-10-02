import io
from pathlib import Path
import argparse
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

URLS = [
    "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv",
    "https://cdn.jsdelivr.net/gh/plotly/datasets@master/heart.csv",
    "https://raw.githubusercontent.com/llSourcell/AI_in_Medicine_Intro/master/heart.csv",
    "https://raw.githubusercontent.com/Ankit152/Heart-Disease-Prediction/master/heart.csv",
]

HEADERS = {"User-Agent": "Mozilla/5.0"}

def _download() -> pd.DataFrame:
    last = None
    for u in URLS:
        try:
            r = requests.get(u, timeout=30, headers=HEADERS)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))
        except Exception as e:
            last = e
            continue
    raise RuntimeError(f"Download failed. Last error: {last}")

def download_data(out_path="data/raw/heart.csv"):
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df = _download()
    df.to_csv(out_path, index=False)
    print(f"Downloaded data to {out_path}")

def split_data(in_path="data/raw/heart.csv"):
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(in_path)
    train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df["target"])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["target"])
    train.to_csv("data/processed/train.csv", index=False)
    val.to_csv("data/processed/val.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)
    print("Saved stratified train/val/test splits in data/processed/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--split", action="store_true")
    args = parser.parse_args()
    if args.download:
        download_data()
    if args.split:
        split_data()
