from dorie import loader

EXPECTED_ALL = [
    "MyDataset",
    "ModelTrainer"
]

def test_imports() -> None:
    assert sorted(loader.__all__) == sorted(EXPECTED_ALL)