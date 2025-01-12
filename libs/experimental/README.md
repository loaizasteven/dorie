# Experimental Directory
This README file provides an overview of the directory structure and key files within the experimental library.

## Structure
- [storage/](storage): Contains modules related to data storage and retrieval.
    - [dump.py](storage/dump.py): Module for dumping data to storage. Currently supports s3 buckets and huggingface hub datasets.
    - [s3_connection.py](storage/s3_connection.py): Module for handling connections to Amazon S3.
    #### AWS S3
    Configuration details can be found [here](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html).

    To use the `s3_connection.py` module, ensure you have the following environment variables set in an `.aws` file:
    - `AWS_ACCESS_KEY_ID`: Your AWS access key ID.
    - `AWS_SECRET_ACCESS_KEY`: Your AWS secret access key.
    - `AWS_DEFAULT_REGION`: The default region to use, e.g., `us-west-2`.

- [synthetic_data/](synthetic_data): Directory for scripts and modules related to generating synthetic data.
    - [base.py](synthetic_data/base.py): Base module for synthetic data generation.
    - [prompts.py](synthetic_data/prompts.py): Module for generating synthetic data prompts.

