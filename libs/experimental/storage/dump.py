from pathlib import Path
import sys
current_dir = Path(__file__).resolve()
sys.path.insert(0, str(current_dir.parent))

from s3_connection import S3Connection
from datasets import load_dataset
import logging
from http import HTTPStatus
from typing import Dict, Union

logger = logging.getLogger(__name__)

@staticmethod
def s3upload(file: str, bucket: str, object_name: str = None) -> str:
    """ Upload the generated file to S3. """
    if bucket:
        s3 = S3Connection()
        response = s3.upload_file(file, bucket, object_name)
        if response.get("statusCode") == HTTPStatus.OK:
            logger.info(f"Success: Synthetic data uploaded to s3 bucket -> {bucket}")
        else:
            logger.error(f"Error: Unable to upload to s3 bucket -> {bucket}")
            logger.error(f"Error: {response.message}")

@staticmethod
def hfupload(path: str, data_file: Union[str, Dict], model_name: str) -> str:
    """ Upload the data file to Hugging Face. """
    dataset = load_dataset(path=path, data_files=data_file)
    dataset.push_to_hub(model_name)
