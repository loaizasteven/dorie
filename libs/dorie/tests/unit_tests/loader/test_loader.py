from dorie.tests import data
from dorie.loader import *

from transformers import BertForSequenceClassification

import pandas

def test_example_data():
    SENTIMATE_CSV = pandas.read_csv(data.SENTIMATE_CSV)
    assert SENTIMATE_CSV.columns.tolist() == ['text', 'label']

def test_transformer():
    # Load Dataset
    dataClass = MyDataset(path=data.SENTIMATE_CSV)
    _ = dataClass.loader()
    print(dataClass)
    print(dataClass.numLabels)
    # Load Model
    trainer = ModelTrainer(
            baseModel='bert-base-uncased', 
            modelArgs={
                'per_device_train_batch_size': 2, 
                'num_train_epochs': 1
                }, 
                device='cpu', 
                data=dataClass
            )
    model = trainer.model
    assert isinstance(model, BertForSequenceClassification)

    
    