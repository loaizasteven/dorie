# Loader of hugging face transformer from /config.json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

import sys
import os

from typing import Any, Optional, Union
from pydantic import BaseModel
from .datatokenizer import MyDataset
from datasets import DatasetDict

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    import warnings
    warnings.warn("torch not found, please install torch to use this module")
    TORCH_AVAILABLE = False
filedir = os.path.dirname(__file__)
sys.path.append(filedir)


class ModelTrainer(BaseModel):
    baseModel: str
    modelArgs: dict
    device: str
    dataClass: MyDataset
    data: Optional[DatasetDict] = None
    model: Optional[AutoModelForSequenceClassification] = None
    tokenizer: Optional[AutoTokenizer] = None
    model_config: Optional[dict] = {'arbitrary_types_allowed': 'true'}

    def __init__(self, baseModel: str, modelArgs: dict, device: str, dataClass: MyDataset, data: DatasetDict = None):
        super().__init__(baseModel=baseModel, modelArgs=modelArgs, device=device, dataClass=dataClass, data=data)
        self.baseModel = baseModel
        self.modelArgs = modelArgs
        self.device = device
        self.dataClass = dataClass
        self.data = data if data else self.dataClass.loader()
        self.model = AutoModelForSequenceClassification.from_pretrained(self.baseModel, num_labels=self.dataClass.numLabels)
        self.tokenizer = AutoTokenizer.from_pretrained(self.baseModel)

        # Model configuration
        self._setdevice()
        self._setlabelmap()

    def train(self):
        training_args = TrainingArguments(**self.modelArgs)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.data['train'],
            eval_dataset=self.data['test'],
            compute_metrics=lambda pred: {'accuracy': (pred.predictions.argmax(-1) == pred.label_ids).mean()}
        )

        trainer.train()
    
    def _setlabelmap(self):
        if hasattr(self.dataClass, 'labelMap'):
            self.model.config.label2id = self.dataClass.labelMap
            self.model.config.id2label = {v: k for k, v in self.model.config.label2id.items()}

    def _setdevice(self):
        if hasattr(torch, self.device):
            device = getattr(torch, self.device)
            set_device = self.device if device.is_available() else 'cpu'
        else:
            warnings.warn(f"Device {self.device} not found. Using default device")
            set_device = 'cpu'

        self.model.to(set_device)

    def save(self, output_dir):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        if TORCH_AVAILABLE:
            with torch.no_grad():
                logits = self.model(**inputs).logits

            predicted_class = torch.argmax(logits)
        else:
            logits = self.model(**inputs).logits
            predicted_class = np.argmax(logits)
        return predicted_class

    def evaluate(self, data: DatasetDict):
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                per_device_eval_batch_size=self.batchSize
            ),
            eval_dataset=data,
            compute_metrics=lambda pred: {'accuracy': (pred.predictions.argmax(-1) == pred.label_ids).mean()}
        )

        return trainer.evaluate()
