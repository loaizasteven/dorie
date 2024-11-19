# Loader of hugging face transformer from /config.json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, AutoConfig
import json

import sys
import os

from pydantic import BaseModel
from .datatokenizer import MyDataset

import torch

filedir = os.path.dirname(__file__)
sys.path.append(filedir)


class ModelTrainer(BaseModel):
    baseModel: str
    modelArgs: dict
    device: str
    data: MyDataset

    def __init__(self, **data):
        super().__init__(**data)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.baseModel, num_labels=data.numLabels)
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
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_class = torch.argmax(logits)
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


if __name__ == '__main__':
    config = json.load(open(f'{filedir}/config.json'))
    trainer = ModelTrainer(**config)
    model = trainer.model()
    print('Model loaded successfully')