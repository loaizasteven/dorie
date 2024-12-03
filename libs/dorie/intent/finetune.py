"""
This module provides functionality for fine-tuning an intent classification model using a custom dataset and configuration.
"""
from intent import Commons, config

from loader import MyDataset, ModelTrainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from typing import Optional, Union
from pydantic import BaseModel

import os.path as osp

import logging

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)    

    console_handler = logging.StreamHandler() # print to console
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger

logger = get_logger('__name__')


class Intent(BaseModel):
    datapath: str 
    config: dict = config()
    dataclass: Optional[MyDataset] = None
    trainer: Optional[Union[ModelTrainer, str]] = None

    def model_post_init(self, *args, **kwargs) -> None:
        """Override this method to perform additional initialization after `__init__` and `model_construct`.
        This is useful if you want to do some validation that requires the entire model to be initialized.
        """
        if isinstance(self.trainer, str):
            logger.info(f"Instance trainer is a string, loading model from path {self.trainer}")
            self._load_local_model(self.trainer)

    def _load_local_model(self, model_path: str) -> None:
        """Load a model from a local path."""
        # TODO: Add to the ModelTrianer class instead?
        self.config['tokenizer'] = AutoTokenizer.from_pretrained(model_path)
        self.config['model'] = AutoModelForSequenceClassification.from_pretrained(model_path)
        
    def load_data(self):
        self.dataclass = MyDataset(path=self.datapath)
        return self.dataclass.loader()

    def train(self):
        data = self.load_data()
        self.trainer = ModelTrainer(**config(), dataClass=self.dataclass, data=data)
        self.trainer.train()

    def save(self, output_dir):
        self.trainer.save(output_dir)

    def push_to_hub(self, model_name: str = 'stevenloaiza/dorie-intent-classifier'):
        model_object = self.config['model'] if isinstance(self.trainer, str) else self.trainer.model
        model_object.push_to_hub(model_name)
        logger.info(f"Model saved to Hugging Face Hub as {model_name}")


if __name__ == '__main__':
    intent_classifier = Intent(
        datapath= osp.join(Commons().fileDir, 'samples/personal-auto-insurance-intents.csv'),
        trainer=osp.join(Commons().fileDir, 'best_results_save')
        )
    # intent_classifier.train()
    # intent_classifier.save('./results')
    # intent_classifier.push_to_hub(model_name='stevenloaiza/dorie-intent-classifier')
