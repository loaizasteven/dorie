from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage

from pydantic import BaseModel
from typing import Dict, Any, Union
from prompts import SYNTHETIC_FEW_SHOT_PREFIX, USER_PROMPT, RESPONSE_FORMAT

import json 


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
    >>> synthdata = CreateData(
    ...         systemprompt = "provide a small finetuning training example",
    ...         userinput = "Provide 1 trianing examples",
    ...         modelname = "gpt-4o-2024-08-06",
    ...         response_format = TrainingClass
    ...     )
    >>> response = synthdata()
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

    def model_post_init(self, __context: Any) -> None:
        """Override this method to perform additional initialization after `__init__` and `model_construct`.
        This is useful if you want to do some validation that requires the entire model to be initialized.
        """
        # Client
        print('Message: initializing client connection')
        self.client = OpenAI()
    
    def invoke(self) -> ChatCompletionMessage:
        self.completion = self.client.beta.chat.completions.parse(
            model = self.modelname,
            max_tokens= self.maxtokens,
            response_format=self.response_format,
            messages=[
                {"role": "system", "content": f"{self.systemprompt}"},
                {"role": "user", "content": f"{self.userinput}"}
            ]
        )

        return self.completion.choices[0].message

    def parseobj(self, classobj:bool =False) -> Union[None, Dict]:
        msg = self.completion.choices[0].message
        return msg.parsed if classobj else json.loads(msg.content)

    def jsondump(self, outputfile:str) -> str:
        try:
            trainingdata = self.parseobj()
            json.dump(trainingdata, open(outputfile, 'w'), indent=4)

            return f"Synthetic data completed and dumped to -> {outputfile}"
        except (AttributeError, BaseException) as e:
            return f"Error: Unable to dump content -> {e} \n"
    
    def __del__(self):
        self.client.close()

    def close(self):
        self.__del__()

        print('Message: closing client connection \n')


if __name__ == "__main__":
    # Generate Call
    synthdata = SyntheticDataGenerator()
    synthdata.invoke()

    # Dump Content
    status = synthdata.jsondump(outputfile = './syntheticinsurancedata.json')
    print(status)

    # close client
    synthdata.close()