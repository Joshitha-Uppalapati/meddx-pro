from pathlib import Path
import argparse
import pandas as pd
from ydata_profiling import ProfileReport

def make_profile(input_csv: str, out_html: str):
    df = pd.read_csv(input_csv)
    prof = ProfileReport(df, title="MedDx-Pro EDA", explorative=True)
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    prof.to_file(out_html)

def feature_dict(input_csv: str, out_md: str):
    df = pd.read_csv(input_csv)
    rows = []
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        missing = int(s.isna().sum())
        unique = int(s.nunique(dropna=True))
        minv = s.min() if pd.api.types.is_numeric_dtype(s) else ""
        maxv = s.max() if pd.api.types.is_numeric_dtype(s) else ""
        rows.append((col, dtype, missing, unique, minv, maxv))
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w") as f:
        f.write("# Feature Dictionary\n\n")
        f.write("| feature | dtype | missing | unique | min | max |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/heart.csv")
    p.add_argument("--profile_html", default="reports/eda_profile.html")
    p.add_argument("--features_md", default="docs/FEATURES.md")
    args = p.parse_args()
    make_profile(args.input, args.profile_html)
    feature_dict(args.input, args.features_md)
    print(f"Wrote {args.profile_html} and {args.features_md}")

if __name__ == "__main__":
    main()
