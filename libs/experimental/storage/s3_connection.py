from pydantic import BaseModel
from typing import Dict, Optional, Any
import logging
import os

import boto3
from botocore.exceptions import ClientError
from http import HTTPStatus


class S3Connection(BaseModel):
    credentials: Optional[Dict[str, str]] = None
    resource: Optional[Any] = None
    client: Optional[Any] = None

    def model_post_init(self, __context: Any) -> None:
        self.resource = boto3.resource("s3")
        self.client = boto3.client("s3")

    def upload_file(self, file_name: str, bucket:str, object_name:str=None) -> Dict:
        # If S3 object_name was not specified, use file_name
        if object_name is None:
            object_name = os.path.basename(file_name)

        # Upload the file
        try:
            logging.info(f"Uploading {file_name} to {bucket}/{object_name}")
            response = self.client.upload_file(file_name, bucket, object_name)
        except ClientError as e:
            logging.error(e)
            return {"statusCode": HTTPStatus.INTERNAL_SERVER_ERROR, "message": str(e)}
        return {"statusCode": HTTPStatus.OK, "message": "File uploaded successfully"}


def check_resources() -> None:
    """
    Use the AWS SDK for Python (Boto3) to create an Amazon Simple Storage Service
    (Amazon S3) resource and list the buckets in your account.
    This example uses the default settings specified in your shared credentials
    and config files.
    """
    s3_resource = boto3.resource("s3")
    print("Hello, Amazon S3! Let's list your buckets:")
    for bucket in s3_resource.buckets.all():
        print(f"\t{bucket.name}")


if __name__ == "__main__":
    check_resources()
