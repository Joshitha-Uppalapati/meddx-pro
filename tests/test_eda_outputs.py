from pathlib import Path
def test_eda_outputs_exist():
    assert Path("reports/eda_profile.html").exists()
    assert Path("docs/FEATURES.md").exists()
