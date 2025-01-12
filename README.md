# <img src="./docs/static/img/Image.jpeg" alt="drawing" width="30"/> DORIE
> Dynamic Omnichannel RoBERTa Intent Engine

*Warning: This is an unstable version of the project. Use with caution as there may be breaking changes until the first stable release (1.0.0).*

## Overview
DORIE (Dynamic Omnichannel RoBERTa Intent Engine) is an advanced natural language processing system designed for multi-channel intent classification. Built on RoBERTa architecture, it provides enterprise-grade NLP capabilities for automated response handling.

## Project Structure
The project is organized as follows:
- `docs/`: Contains documentation files and static assets.
- `libs/`: Includes library scripts and utilities.
    - `dorie/`: Core library for DORIE functionalities.
        - `virtualenv.sh`: Script to set up a virtual environment.
        - `hf_management.sh`: Script for managing Huggingface cache.
- `scripts/`: Contains various scripts for data processing and model training.
- `tests/`: Unit and integration tests for the project.
- `requirements.txt`: List of dependencies for setting up the project using virtualenv.

## Roadmap
- **Version 1.0.0**: First stable release with core functionalities.
- **Future Enhancements**:
    - Enhanced model training pipelines.
    - Integration with more databases and data structures.
    - Include LoRA (Low-Rank Adaptation) for efficient fine-tuning.
    - Improved multi-language support.
    - Advanced error handling and logging mechanisms.
    - User-friendly configuration and setup process.
    - Comprehensive documentation and tutorials.
    - Additional cookbook examples and pytest
    - Updates to huggingface modelcard

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

## Contributing
We welcome contributions from the community. Please refer to the `CONTRIBUTING.md` (TBD) file for guidelines on how to contribute to the project.