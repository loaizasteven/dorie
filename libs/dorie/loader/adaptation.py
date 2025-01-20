#  ------------------------------------------------------------------------------------------
#  Leverage huggingface Parameter-Efficient Fine Tuning Library
#  Default behavior utilizes LORA: Low Rank Adaptation Method for fine tuning
#  ------------------------------------------------------------------------------------------
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

import logging

logger = logging.getLogger(__name__)

def return_peft_model(
        model_name_or_path:str, 
        lora_config:dict = {
            "task_type": TaskType.SEQ_CLS, 
            "inference_mode": False, 
            "r": 8, 
            "lora_alpha": 32, 
            "lora_dropout": 0.1
        }, 
        num_labels:int=2,
        verbose:bool=False
    ):
    """Return the PEFT model"""
    peft_config = LoraConfig(**lora_config)

    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
    model = get_peft_model(model, peft_config)
    if verbose:
        logger.info(model.print_trainable_parameters())
    
    return model


if __name__ == "__main__":
    return_peft_model("bert-base-uncased", "bert-base-uncased")