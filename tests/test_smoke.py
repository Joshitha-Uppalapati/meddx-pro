from pathlib import Path

def test_data_files_exist():
    assert Path("data/raw/heart.csv").exists()
    assert Path("data/processed/train.csv").exists()
    assert Path("data/processed/val.csv").exists()
    assert Path("data/processed/test.csv").exists()
