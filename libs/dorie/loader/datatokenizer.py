# Preprcess the dataset and tokenize the input sentence
from datasets import Dataset, DatasetDict
from datasets.formatting.formatting import LazyBatch
from transformers import AutoTokenizer

from pydantic import BaseModel

import pandas as pd

import os
import sys
from typing import Optional, Union

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  

class MyDataset(BaseModel):
    path: str
    labelmap: Optional[dict] = None
    split: float = 0.8
    pretrained_model_name: str = 'roberta-base'
    num_labels: Optional[int] = None

    def _setnumlabels(self, dataset: Dataset):
        """Set the number of labels in the dataset"""
        self.num_labels = len(dataset['label'].unique())

    def _fileformat(self):
        """Return the file format of the dataset"""
        return self.path.split('.')[-1]
    
    def _csvconverter(self, path: str):
        """Convert CSV dataset to DatasetDict"""
        dataset = pd.read_csv(path)
        self._setnumlabels(dataset)
        label_map = self.labelmap or {label: i for i, label in enumerate(dataset['label'].unique())}
        dataset['label'] = dataset['label'].map(label_map)

        data_dict = {'text': dataset['text'].tolist(), 'label': dataset['label'].tolist()}
        split_idx = int(self.split * len(data_dict['text']))

        return DatasetDict({
            'train': Dataset.from_dict({'text': data_dict['text'][:split_idx], 'label': data_dict['label'][:split_idx]}),
            'test': Dataset.from_dict({'text': data_dict['text'][split_idx:], 'label': data_dict['label'][split_idx:]})
        })

    def _jsonlconverter(self, path:str):
        """Convert JSONL dataset to DatasetDict"""
        pass

    def loader(self, format:str = 'torch'):
        """Load the dataset"""
        mapping = {'csv': self._csvconverter, 'jsonl': self._jsonlconverter}
        converter = mapping.get(self._fileformat())
        assert converter, f"File format {self._fileformat()} not supported"

        dataset = converter(self.path)

        dataset = dataset.map(self.preprocess, batched=True)
        dataset.set_format(format)

        return dataset
    
    def preprocess(self, examples: LazyBatch,  max_length: int = 128):
        """Preprocess the input sentence"""
        tokenizer = self.tokenizer()

        tokenized_examples = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
        tokenized_examples['label'] = examples['label']
        return tokenized_examples

    def tokenizer(self):
        """Load the tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        return tokenizer

    def __call__(self, *args, **kwds):
        return self.loader(*args, **kwds)
    
    def invoke(self, *args, **kwds):
        return self.__call__(*args, **kwds)
    