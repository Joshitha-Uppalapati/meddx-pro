from pathlib import Path

def test_m4_outputs_exist():
    ok = (
        Path("artifacts/top_features.json").exists() and
        Path("artifacts/fairness_summary.json").exists() and
        (Path("reports/shap_beeswarm.png").exists() or Path("reports/perm_importance.png").exists()) and
        Path("reports/fairness_report.md").exists()
    )
    assert ok
