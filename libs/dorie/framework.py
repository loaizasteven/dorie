from typing import Optional, List , Any
from pydantic import BaseModel, Field

from loader import MyDataset, tokenizer
from datasets import DatasetDict

from pathlib import Path
import sys
import logging
import warnings

sys.path.append(str(Path(__file__).parents[2]))
from utils.cli import clap

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class LUFramework(BaseModel):
    config: Any = None

    def model_post_init(self, *args, **kwargs) -> None:
        """CLI Information for user"""
        if __name__ == "__main__":
            logger.info(
                "\033[92m" + """
                Welcome to the Language Understanding Framework! This is a simple framework utilized to fine-tune a downstream tasks
                from a pre-trained based model. The framework is built on top of HuggingFace's Transformers library and PyTorch. In the 
                future, we will add support for continual pre-trianing (using DAPT and TAPT) as well as an evaluation framework starting 
                with CLS tasks.  
            """ + "\033[0m" 
            )

    def _load_dataset(self, format: str = "torch") -> DatasetDict:
        """Load the dataset using MyDataset class"""
        if self.config.datapath.endswith(".csv"):
            logger.info(f"Loading local file from {self.config.datapath}")
            dataclass = MyDataset(
                path=self.config.datapath, 
                split=self.config.split
                )
            dataset = dataclass._csvconverter(path=self.config.datapath)
            dataset = dataset.map(dataclass.preprocess, batched=True)

            try:
                dataset.set_format(format)
            except:
                warnings.warn(f"Format {format} not supported. Using default format")
            
            return dataset

        else:
            logger.info(f"Attempting to load dataset from HuggingFace local cache `~/.cache/huggingface` otherwise will load from Dataset Hub, searching for '{self.config.datapath}'")
            dataclass = MyDataset(path=self.config.datapath)
            return dataclass.loader()

    def _create_training_args(self):
        """Generate [Dict] training arguments. Opting to not use proper JSON naming conventions 
            for ease of unwrapping the arguments."""

        return {
            "output_dir": self.config.output_dir,
            "learning_rate": self.config.learning_rate,
            "num_train_epochs": self.config.num_train_epochs,
            "per_device_train_batch_size": self.config.per_device_train_batch_size,
            "per_device_eval_batch_size": self.config.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "warmup_steps": self.config.warmup_steps,
            "weight_decay": self.config.weight_decay,
            "logging_steps": self.config.logging_steps,
            "evaluation_strategy": self.config.evaluation_strategy,
            "eval_steps": self.config.eval_steps,
            "save_strategy": self.config.save_strategy,
            "save_steps": self.config.save_steps,
            "fp16": self.config.fp16
        }
    
    def invoke(self) -> None:
        """Main method to invoke the LUFramework"""
        data_object = self._load_dataset()
        print(data_object)


@clap
class LUFrameworkArgs(BaseModel):
    model: str = Field(..., description="Name or path of the base model")
    output_dir: str = Field(..., description="Directory to save the fine-tuned model")
    datapath: str = Field(..., description="Path to the dataset or name to huggingface datasets")
    split: float = Field(default=0.8, description="Train-test split ratio for local csv datasets")
    lora_r: int = Field(default=8, description="LoRA attention dimension")
    lora_alpha: int = Field(default=16, description="LoRA alpha parameter")
    lora_dropout: float = Field(default=0.1, description="LoRA dropout rate")
    learning_rate: float = Field(default=2e-5, description="Learning rate for training")
    num_train_epochs: int = Field(default=3, description="Number of training epochs")
    per_device_train_batch_size: int = Field(default=8, description="Training batch size per device")
    per_device_eval_batch_size: int = Field(default=8, description="Evaluation batch size per device")
    gradient_accumulation_steps: int = Field(default=1, description="Number of gradient accumulation steps")
    warmup_steps: int = Field(default=0, description="Number of warmup steps")
    weight_decay: float = Field(default=0.01, description="Weight decay")
    logging_steps: int = Field(default=10, description="Number of steps between logging")
    evaluation_strategy: str = Field(default="steps", description="Evaluation strategy")
    eval_steps: int = Field(default=50, description="Number of steps between evaluations")
    save_strategy: str = Field(default="steps", description="Save strategy")
    save_steps: int = Field(default=50, description="Number of steps between saves")
    fp16: bool = Field(default=False, description="Use mixed precision training")
    verbose: bool = Field(default=False, description="Verbose logging")


if __name__ == "__main__":
    config = LUFrameworkArgs()
    if config.verbose:
        logging.basicConfig(level=logging.DEBUG)

    lu_framework = LUFramework(config=config)
    lu_framework.invoke()