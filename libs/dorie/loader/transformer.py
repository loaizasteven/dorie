# Loader of hugging face transformer from /config.json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, AutoConfig
import json

import sys
import os

filedir = os.path.dirname(__file__)
sys.path.append(filedir)

config = json.load(open(f'{filedir}/config.json'))

def load_model():
    "Load pre-trained model and tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(config['baseModel'])
    model = AutoModelForSequenceClassification.from_pretrained(config['baseModel'])
    model.to(config['device'])
    return model, tokenizer

if __name__ == '__main__':
    model, tokenizer = load_model()
    print(model)
    print(tokenizer)
    print(config)
    print('Model loaded successfully')