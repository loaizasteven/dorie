# <img src="./docs/static/img/Image.jpeg" alt="drawing" width="30"/> DORIE
> Dynamic Omnichannel RoBERTa Intent Engine

## Overview
DORIE (Dynamic Omnichannel RoBERTa Intent Engine) is an advanced natural language processing system designed for multi-channel intent classification. Built on RoBERTa architecture, it provides enterprise-grade NLP capabilities for automated response handling.


## Getting Started

### Prerequisites
- Python 3.8+
- Huggingface account
- Poetry or virtualenv

### Huggingface Setup
1. Create an account at huggingface.co
2. Set up authentication:
```bash
huggingface-cli login
```
Credentials will be stored in `/Users/<USERNAME>/.cache/huggingface/stored_tokens`

### Installation Options

#### Using Poetry (Recommended)
```bash
# Install Poetry
brew install pipx
pipx install poetry
pipx ensurepath
pipx upgrade poetry

# Initialize project
poetry install
```

Use the environment:
- Run scripts: `poetry run python script.py`
- Activate shell: `poetry shell`

#### Using Virtualenv
Due to torch compatibility issues with Poetry, you can alternatively:
1. Use the [virtualenv script](./libs/dorie/virtualenv.sh)
2. Install dependencies from [requirements.txt](./requirements.txt)

### Cache Management
For Huggingface cache management, refer to [hf_management.sh](./libs/dorie/hf_management.sh).