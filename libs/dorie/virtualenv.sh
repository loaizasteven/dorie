#!/usr/bin/python

REPO="venv"
REPONAME="dorie"
DIRECTORY="dorie_env"

function activate () {
  source $HOME/git/$REPO/$DIRECTORY/bin/activate
  # logging info
  echo "Currently activated $VIRTUAL_ENV"
  python --version
  # pip installation
  pip install --upgrade pip
  echo "pip installation in quiet mode..."
  pip install -q -r $HOME/git/$REPONAME/requirements.txt --upgrade-strategy only-if-needed
}

if [ -d "$HOME/git/$REPO/$DIRECTORY" ]; then
    activate
else
    echo "Creating venv"
    /usr/local/bin/python3 -m venv "$HOME/git/$REPO/$DIRECTORY"

    activate
fi