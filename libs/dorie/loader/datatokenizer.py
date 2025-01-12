# Preprcess the dataset and tokenize the input sentence
from datasets import Dataset, DatasetDict, load_dataset
from datasets.formatting.formatting import LazyBatch
from transformers import AutoTokenizer
import datasets.exceptions as dataset_exceptions
from pydantic import BaseModel

import pandas as pd

import os
import sys
from typing import Optional, Union
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  

class MyDataset(BaseModel):
    path: Union[str, Path]
    split: float = 0.8
    pretrained_model_name: str = 'roberta-base'
    numLabels: Optional[int] = None
    labelMap: Optional[dict] = None

    def _setnumlabels(self, dataset: Dataset):
        """Set the number of labels in the dataset"""
        self.numLabels = len(dataset['label'].unique())

    def _fileformat(self):
        """Return the file format of the dataset"""
        return self.path.split('.')[-1] if isinstance(self.path, str) else self.path.suffix[1:]
    
    def _setlabelmap(self, dataset: Dataset) -> None:
        """Set the label map"""
        self.labelMap = self.labelMap or {str(label): i for i, label in enumerate(dataset['label'].unique())}

    def _csvconverter(self, path: str):
        """Convert CSV dataset to DatasetDict"""
        dataset = pd.read_csv(path)
        self._setnumlabels(dataset)
        self._setlabelmap(dataset=dataset)

        # TODO: Add support for dumping label map to a file/instance
        # dataset['label'] = dataset['label'].map(self.labelMap)

        data_dict = {'text': dataset['text'].tolist(), 'label': dataset['label'].tolist()}
        split_idx = int(self.split * len(data_dict['text']))

        return DatasetDict({
            'train': Dataset.from_dict({'text': data_dict['text'][:split_idx], 'label': data_dict['label'][:split_idx]}),
            'test': Dataset.from_dict({'text': data_dict['text'][split_idx:], 'label': data_dict['label'][split_idx:]})
        })

    def _jsonlconverter(self, path:str):
        """Convert JSONL dataset to DatasetDict"""
        pass

    def _hfhub(self):
        """Load the dataset from HuggingFace Hub"""
        train, test = load_dataset(self.path, split=['train', 'test'])
        self._setnumlabels(train.data)
        self._setlabelmap(dataset=train.data)

        return DatasetDict({
            "train": train,
            "test": test
        })

    def loader(self, format:str = 'torch'):
        """Load the dataset"""
        try: 
            dataset = self._hfhub()
        except (dataset_exceptions.DatasetNotFoundError, FileNotFoundError):
            mapping = {'csv': self._csvconverter, 'jsonl': self._jsonlconverter}
            converter = mapping.get(self._fileformat(), self._hfhub)
            assert converter, f"File format {self._fileformat()} not supported"

            dataset = converter(self.path)
        dataset = dataset.map(self.preprocess, batched=True)
        try:
            dataset.set_format(format)
        except:
            import warnings
            warnings.warn(f"Format {format} not supported. Using default format")

        return dataset
    
    def preprocess(self, examples: LazyBatch,  max_length: int = 128):
        """Preprocess the input sentence"""
        tokenizer = self.tokenizer()

        tokenized_examples = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
        tokenized_examples['label'] = [self.labelMap[label] for label in examples['label']]
        return tokenized_examples

    def tokenizer(self):
        """Load the tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        return tokenizer

    def __call__(self, *args, **kwds):
        return self.loader(*args, **kwds)
    
    def invoke(self, *args, **kwds):
        return self.__call__(*args, **kwds)
