SYNTHETIC_FEW_SHOT_PREFIX = '''
    If you do not provide 100 random examples for the following task, you will fail!
    Generate synthetic data for a personal auto car insurance intent such as saleQuote, addDriver, payPrem, 
    startClaim. You are restricted to the labels provided in this example, the text should be conversation utterance
    for interactions between a customer and an insurance agent. Additionally, include some examples that are not related to the labels provided but still potential insurance related utterances, these examples should be classified as 'noIntent'.
    '''

USER_PROMPT = "YOU MUST PROVIDE 100 EXAMPLES, DO NOT FAIL!"

RESPONSE_FORMAT = {
        "type": "json_schema", 
        "json_schema":{
            "name": "syntheticdata", 
            "strict": True, 
            "schema": {
                "type": "object", 
                "properties": {
                    "label": {"type": "array", "items": {"type": "string"}}, 
                    "text": {"type": "array", "items": {"type": "string"}}
                },
                "additionalProperties": False,
                "required": ["label", "text"]
                }
            }
        }