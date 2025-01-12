SYNTHETIC_FEW_SHOT_PREFIX = '''
    If you do not provide 2000 random examples for the following task, you will fail!
    Generate synthetic data for a personal auto car insurance intent such as saleQuote, addDriver, payPrem, 
    startClaim. You are restricted to the labels provided in this example, the text should be conversation utterance
    for interactions between a customer and an insurance agent.
    '''

USER_PROMPT = "YOU MUST PROVIDE 2000 EXAMPLES, DO NOT FAIL!"

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