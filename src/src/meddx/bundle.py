import json, joblib, time
from pathlib import Path

BUNDLE_PATH = Path("artifacts/bundle.joblib")
META_PATH = Path("artifacts/metadata.json")

def save_bundle(model, threshold=0.5, extra=None):
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": float(threshold)}, BUNDLE_PATH)
    meta = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "threshold": float(threshold)}
    if extra: meta.update(extra)
    with open(META_PATH, "w") as f: json.dump(meta, f, indent=2)
    return str(BUNDLE_PATH), str(META_PATH)

def load_bundle():
    obj = joblib.load(BUNDLE_PATH)
    model = obj["model"]
    thr = float(obj.get("threshold", 0.5))
    meta = {}
    if META_PATH.exists():
        with open(META_PATH) as f: meta = json.load(f)
    return model, thr, meta
