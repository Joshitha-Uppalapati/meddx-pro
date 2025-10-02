from pathlib import Path
import json

def test_artifacts_exist():
    assert Path("artifacts/model.joblib").exists()
    assert Path("artifacts/val_metrics.json").exists()
    assert Path("artifacts/test_metrics.json").exists()
    assert Path("reports/roc_val.png").exists()
    assert Path("reports/pr_val.png").exists()

def test_metrics_keys():
    data = json.loads(Path("artifacts/val_metrics.json").read_text())
    assert "best_model" in data and "metrics" in data
