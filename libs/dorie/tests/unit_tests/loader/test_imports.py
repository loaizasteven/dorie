from dorie import loader

EXPECTED_ALL = [
    "MyDataset",
    "ModelTrainer",
    "tokenizer"
]

def test_imports() -> None:
    assert sorted(loader.__all__) == sorted(EXPECTED_ALL)