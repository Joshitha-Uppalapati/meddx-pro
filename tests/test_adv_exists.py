from pathlib import Path

def test_adv_artifacts():
    assert Path("artifacts/model_adv.joblib").exists()
    assert Path("artifacts/adv_metrics.json").exists()

def test_adv_eval_outputs():
    assert Path("reports/calibration_curve.png").exists()
    assert Path("artifacts/adv_test_metrics.json").exists()
