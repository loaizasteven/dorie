from intent import Commons, config
"""
This module provides functionality for fine-tuning an intent classification model using a custom dataset and configuration.
"""
from loader import MyDataset, ModelTrainer

from typing import Optional
from pydantic import BaseModel

import os.path as osp


class Intent(BaseModel):
    datapath: str 
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


if __name__ == '__main__':
    intent_classifier = Intent(datapath= osp.join(Commons().fileDir, 'samples/personal-auto-insurance-intents.csv'))
    intent_classifier.train()
    intent_classifier.save('./results')
