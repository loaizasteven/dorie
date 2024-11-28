import sys
from pathlib import Path

from pydantic import BaseModel

import json
import os

_THIS_DIR = Path(__file__).parent
_PARENT_DIR = _THIS_DIR.parent


class Commons(BaseModel):
    fileDir: str = str(_THIS_DIR)
    parentDir: str = str(_PARENT_DIR)

sys.path.insert(0, Commons().parentDir)
sys.path.insert(0, Commons().fileDir)

class Config(BaseModel):
    fileDir: str = str(Commons().fileDir)
    
    def _traverseloadconfig(self):

        config_file = None
        for root, dirs, files in os.walk(self.fileDir):
            for file in files:
                if file.endswith("config.json"):
                    config_file = Path(root) / file
                    break
            if config_file:
                break

        if config_file:
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError("No config.json file found in the directory.")

    def __call__(self):
        return self._traverseloadconfig()
    
def config():
    conf = Config()
    return conf._traverseloadconfig()