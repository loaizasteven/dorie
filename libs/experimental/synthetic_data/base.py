from openai import AsyncOpenAI as OpenAI
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage

from pydantic import BaseModel
from typing import Dict, Any, Union

from pathlib import Path
import sys
current_dir = Path(__file__).resolve()
sys.path.insert(0, str(current_dir.parent))

from prompts import SYNTHETIC_FEW_SHOT_PREFIX, USER_PROMPT, RESPONSE_FORMAT

import json 
import logging
import csv
import io
import asyncio
import uuid

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SyntheticDataGenerator(BaseModel):
    """ Class to generate sample finetuning data from LLM to be used for task specific SLM

    Docstring Test
    >>> from pydantic import BaseModel
    >>> from typing import Dict, List
    >>> class Message(BaseModel):
    ...     role: str
    ...     content: str
    >>> class OpenAIResponse(BaseModel):
    ...     messages: List[Message]
    >>> class TrainingClass(BaseModel):
    ...     data: List[OpenAIResponse]
    >>> synthdata = SyntheticDataGenerator(
    ...         systemprompt = "provide a small finetuning training example",
    ...         userinput = "Provide 1 trianing examples",
    ...         modelname = "gpt-4o-2024-08-06",
    ...         response_format = TrainingClass
    ...     )
    >>> response = synthdata.invoke()
    >>> isinstance(response.parsed, TrainingClass)
    True

    Attributes:
        systemprompt: system message prompt
        userinput: input query
        modelname: model name, see OpenAi for available models
        client: OpenAI client
        maxtokens: The maximum number of tokens to generate includes both the prompt and completion, has a ceiling based
                    on model choice.
        completion: ChatCompletion type generate within the __call__() method
    """
    systemprompt: str = SYNTHETIC_FEW_SHOT_PREFIX
    userinput: str = USER_PROMPT
    modelname: str = "gpt-4o-2024-08-06"
    client: Any | None = None
    maxtokens: int = 16000
    completion: ChatCompletion | None = None
    response_format: Dict | Any = RESPONSE_FORMAT
    sessionid: uuid.UUID | None = None

    def model_post_init(self, __context: Any) -> None:
        """Override this method to perform additional initialization after `__init__` and `model_construct`.
        This is useful if you want to do some validation that requires the entire model to be initialized.
        """
        self.sessionid = uuid.uuid4()
        if not self.client:
            logging.debug('Message: initializing client connection')
            self.client = OpenAI()
    
    async def invoke(self) -> ChatCompletionMessage:
        logger.info(f"Generating synthetic data; sessionId:{self.sessionid}")
        self.completion = await self.client.beta.chat.completions.parse(
            model = self.modelname,
            max_tokens= self.maxtokens,
            response_format=self.response_format,
            messages=[
                {"role": "system", "content": f"{self.systemprompt}"},
                {"role": "user", "content": f"{self.userinput}"}
            ]
        )
        logger.info(f"Completed; sessionId:{self.sessionid}")
        return self.completion.choices[0].message

    def parseobj(self, classobj:bool =False) -> Union[None, Dict]:
        msg = self.completion.choices[0].message
        return msg.parsed if classobj else json.loads(msg.content)
    
    def _json_to_csv(self, jsondata:Dict) -> str:
        """Convert json data to csv format, using io.StringIO to write to memory"""
        csvfile = io.StringIO()
        csvwriter = csv.writer(csvfile)

        # Write the headers/data
        csvwriter.writerow(jsondata.keys())
        csvwriter.writerows(zip(*jsondata.values()))

        csvfile.seek(0) # reset the file pointer
        return csvfile.getvalue()
    
    def save(self, outputfile: str, format: str = 'csv') -> str:
        """Save the synthetic data to a file"""
        trainingdata = self.parseobj()
        if format == 'json':
            return self.jsondump(trainingdata, outputfile)
        else:
            return self.csvdump(trainingdata, outputfile)

    def jsondump(self, object: dict, outputfile: str) -> str:
        try:
            with open(outputfile, 'w') as f:
                json.dump(object, f, indent=4)
            return f"Success: Synthetic data completed and dumped to -> {outputfile}"
        except (AttributeError, BaseException) as e:
            return f"Error: Unable to dump content -> {e} \n"

    def csvdump(self, object: dict, outputfile: str) -> str:
        try:
            csv_content = self._json_to_csv(object)
            with open(outputfile, 'w') as f:
                f.write(csv_content)
            return f"Success: Synthetic data completed and dumped to -> {outputfile}"
        except (AttributeError, BaseException) as e:
            return f"Error: Unable to dump content -> {e} \n"
        
    async def close(self):
        logger.info(f"Closing client connection Object_{self.sessionid}")
        await self.client.close()


if __name__ == "__main__":
    async def test_corutine_syntheticdata():
        synthdata = SyntheticDataGenerator()
        synthdata2 = SyntheticDataGenerator(systemprompt = "provide a small finetuning training example", userinput = "Provide 1 trianing examples",)
        await asyncio.gather(synthdata.invoke(), synthdata2.invoke())

        status = synthdata.save(outputfile = './syntheticinsurancedata.csv')
        print(status)

        await synthdata2.close()

    # Generate concurrent openai calls 
    asyncio.run(test_corutine_syntheticdata())
