from intent import Commons, config
"""
This module provides functionality for fine-tuning an intent classification model using a custom dataset and configuration.

Classes:
    Intent(BaseModel): A class for managing the dataset, model training, and saving the trained model.

Functions:
    load_data(self): Loads the dataset using the specified datapath and initializes the dataclass attribute.
    train(self): Trains the model using the loaded dataset and configuration.
    save(self, output_dir): Saves the trained model to the specified output directory.

Usage:

    # Load RoBERTa-base model for sequence classification

    sample_text = 'I want to pay my premium in full'

Notes:
    - `return_tensors="pt"` in the tokenizer function indicates that the tokenized inputs should be returned as PyTorch tensors.
"""
from loader import MyDataset, ModelTrainer

from typing import Optional
from pydantic import BaseModel

import os.path as osp


class Intent(BaseModel):
    datapath: str = osp.join(Commons().fileDir, 'samples/personal-auto-insurance-intents.csv')
    config: dict = config()
    dataclass: Optional[MyDataset] = None
    trainer: Optional[ModelTrainer] = None

    def load_data(self):
        self.dataclass = MyDataset(path=self.datapath)
        print('hihi', type(self.dataclass))
        return self.dataclass.loader()

    def train(self):
        data = self.load_data()
        self.trainer = ModelTrainer(**config(), dataClass=self.dataclass, data=data)
        self.trainer.train()

    def save(self, output_dir):
        self.trainer.save(output_dir)

intent_classifier = Intent()
intent_classifier.train()
intent_classifier.save('./results')
