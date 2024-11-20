# Loader of hugging face transformer from /config.json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

import sys
import os

from typing import Any, Optional
from pydantic import BaseModel
from .datatokenizer import MyDataset

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
    data: MyDataset
    model: Optional[AutoModelForSequenceClassification] = None
    tokenizer: Optional[AutoTokenizer] = None
    model_config: Optional[dict] = {'arbitrary_types_allowed': 'true'}

    def __init__(self, baseModel: str, modelArgs: dict, device: str, data: MyDataset):
        super().__init__(baseModel=baseModel, modelArgs=modelArgs, device=device, data=data)
        self.baseModel = baseModel
        self.modelArgs = modelArgs
        self.device = device
        self.data = data
        self.model = AutoModelForSequenceClassification.from_pretrained(self.baseModel, num_labels=self.data.numLabels)
        self.tokenizer = AutoTokenizer.from_pretrained(self.baseModel)

    def train(self, data):
        training_args = TrainingArguments(**self.modelArgs)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=data['train'],
            eval_dataset=data['test'],
            compute_metrics=lambda pred: {'accuracy': (pred.predictions.argmax(-1) == pred.label_ids).mean()}
        )

        trainer.train()

    def save(self):
        self.model.save_pretrained(self.outputDir)
        self.tokenizer.save_pretrained(self.outputDir)

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

    def evaluate(self, data):
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                per_device_eval_batch_size=self.batchSize
            ),
            eval_dataset=data,
            compute_metrics=lambda pred: {'accuracy': (pred.predictions.argmax(-1) == pred.label_ids).mean()}
        )

        return trainer.evaluate()
