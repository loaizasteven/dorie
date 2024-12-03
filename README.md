# <img src="./docs/static/img/Image.jpeg" alt="drawing" width="30"/> DORIE
> Dynamic Omnichannel RoBERTa Intent Engine

## Huggingface Hub
Set up token from huggingface.co and run the following on the terminal `huggingface login` to set your credentials in a cache (e.g. `/Users/<USERNAME>/.cache/huggingface/stored_tokens`)

## Python Dependency Management
This project uses [python-poetry](https://python-poetry.org) for python packages and dependencies. 

Initial Installation and project set up
```bash
brew install pipx
pipx  install poetry
pipx ensurepath 
pipx upgrade poetry
```

Requires a new terminal to reflect changes
```bash
poetry init # Existing Project
```

To use the env use `poetry run python <>.py` or activate the shell `poetry shell`, the latter will activate the virtual enviornment from cache dir.

**Note**: Current issue with installing `torch` using poetry, use [venv shell script](./libs/dorie/virtualenv.sh) and requirements [file](./requirements.txt).