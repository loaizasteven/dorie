from openai import AsyncOpenAI as OpenAI
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage

from pydantic import BaseModel
from typing import Dict, Any, Union
import os

import random

from pathlib import Path
import sys
current_dir = Path(__file__).resolve()
sys.path.insert(0, str(current_dir.parent))
sys.path.insert(0, str(current_dir.parents[1]))

from storage.dump import hfupload, s3upload
from prompts import SYNTHETIC_FEW_SHOT_PREFIX, USER_PROMPT, RESPONSE_FORMAT

from datasets import load_dataset

import json 
import logging
import csv
import io
import asyncio
import uuid
from http import HTTPStatus

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SyntheticDataGenerator(BaseModel):
    """ Class to generate sample finetuning data from LLM to be used for task-specific SLM """

    systemprompt: str = SYNTHETIC_FEW_SHOT_PREFIX
    userinput: str = USER_PROMPT  # Explicit instruction to provide 100 examples
    modelname: str = "gpt-4o"
    client: Any | None = None
    maxtokens: int = 16000  # Ensure this is large enough for the entire response
    completion: ChatCompletion | None = None
    response_format: Dict | Any = RESPONSE_FORMAT
    sessionid: uuid.UUID | None = None

    def model_post_init(self, __context: Any) -> None:
        """ Override to perform additional initialization after `__init__` and `model_construct`. """
        self.sessionid = uuid.uuid4()
        if not self.client:
            logging.debug('Message: initializing client connection')
            self.client = OpenAI()

    async def invoke(self) -> Dict:
        logger.info(f"Generating synthetic data; sessionId:{self.sessionid}")
        completion = await self.client.beta.chat.completions.parse(
            model=self.modelname,
            max_tokens=self.maxtokens,
            response_format=self.response_format,
            messages=[
                {"role": "system", "content": f"{self.systemprompt}"},
                {"role": "user", "content": f"{self.userinput}"}
            ]
        )
        logger.info(f"Completed; sessionId:{self.sessionid}")
        return self.parseobj(completion)

    def parseobj(self, parsed_response, classobj: bool = False) -> Union[None, Dict]:
        """ Parse the response into usable data format. """
        msg = parsed_response.choices[0].message
        return msg.parsed if classobj else json.loads(msg.content)

    def _json_to_csv(self, jsondata: Dict) -> str:
        """ Convert JSON data to CSV format. """
        csvfile = io.StringIO()
        csvwriter = csv.writer(csvfile)

        # Write the headers/data
        csvwriter.writerow(jsondata.keys())
        csvwriter.writerows(zip(*jsondata.values()))

        csvfile.seek(0)  # Reset the file pointer
        return csvfile.getvalue()

    def save(self, trainingdata: dict, outputfile: str, format: str = 'csv') -> str:
        """ Save the synthetic data to a file. """
        if format == 'json':
            return self.jsondump(trainingdata, outputfile)
        else:
            return self.csvdump(trainingdata, outputfile)

    def jsondump(self, object: dict, outputfile: str) -> str:
        """ Save JSON data to file. """
        try:
            with open(outputfile, 'w') as f:
                json.dump(object, f, indent=4)
            return f"Success: Synthetic data completed and dumped to -> {outputfile}"
        except (AttributeError, BaseException) as e:
            return f"Error: Unable to dump content -> {e} \n"

    def csvdump(self, object: dict, outputfile: str) -> str:
        """ Save CSV data to file. """
        try:
            csv_content = self._json_to_csv(object)
            with open(outputfile, 'w') as f:
                f.write(csv_content)
            logger.info(f"Success: Synthetic data completed and dumped to -> {outputfile}")
        except (AttributeError, BaseException) as e:
            logger.info(f"Error: Unable to dump content -> {e} \n")

    async def close(self):
        """ Close the client connection. """
        logger.info(f"Closing client connection Object_{self.sessionid}")
        await self.client.close()

def merge_dicts(dict_list):
    merged_dict = {}
    for d in dict_list:
        for key, value in d.items():
            if key not in merged_dict:
                merged_dict[key] = []
            merged_dict[key].extend(value)
    return merged_dict

def dict_split(dict_, split=0.8):
    train = {'label': [], 'text': []}
    test = {'label': [], 'text': []}
    
    # Combine 'label' and 'text' into a list of tuples for shuffling
    combined = list(zip(dict_['label'], dict_['text']))
    
    # Shuffle the combined list
    random.shuffle(combined)
    
    # Calculate the split index
    split_index = int(len(combined) * split)
    
    # Split the data into train and test
    train_combined = combined[:split_index]
    test_combined = combined[split_index:]
    
    # Unzip the combined lists back into 'label' and 'text'
    train['label'], train['text'] = zip(*train_combined)
    test['label'], test['text'] = zip(*test_combined)
    
    # Convert the tuples back to lists
    train['label'] = list(train['label'])
    train['text'] = list(train['text'])
    test['label'] = list(test['label'])
    test['text'] = list(test['text'])
    
    return train, test


async def corutine_syntheticdata(n: int = 1, outputfile: str = 'syntheticdata'):
    """ Coroutine to generate synthetic data 
    Limitation:
     You may encounter `openai.RateLimitError` if too many concurrent requests are made.
    """
    synthdata_instances = [SyntheticDataGenerator() for _ in range(n)]
    response = await asyncio.gather(*[synthdata.invoke() for synthdata in synthdata_instances])
    if n == 1:
        len(response[0]['label'])
        train_flatten, test_flatten = dict_split(response[0])
    else:
        split = int(n * 0.8)
        train_flatten = merge_dicts(response[:-split])
        test_flatten = merge_dicts(response[-split:])

    logger.info(f"""
                Corutine returned:
                    Trianing Examples: {len(train_flatten['label'])}
                    Testing Examples: {len(test_flatten['label'])}""
                """)
    # Save the synthetic data to a file
    synthdata_instances[0].csvdump(train_flatten, f"{outputfile}_train.csv")
    synthdata_instances[0].csvdump(test_flatten, f"{outputfile}_test.csv")


    # hfupload(
    #     path='csv', 
    #     data_files={
    #         "train": f"{outputfile}_train.csv", 
    #         "test": f"{outputfile}_test.csv"
    #     }, 
    #     model_name='stevenloaiza/synthetic_insurance_data')


if __name__ == "__main__":
    asyncio.run(corutine_syntheticdata())
