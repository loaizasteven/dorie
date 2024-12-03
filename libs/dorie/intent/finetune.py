from intent import Commons, config
"""
This module provides functionality for fine-tuning an intent classification model using a custom dataset and configuration.
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
