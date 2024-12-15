SYNTHETIC_FEW_SHOT_PREFIX = '''
    Im fine tuning roBERTa for intent (text multi classification), creating a training file for 
    personal auto car insurance intent (sale, service, claims) such as saleQuote, addDriver, payPrem, 
    startClaim. You are restricted to the labels provided in this example, the text should be conversation utterance
    for interactions between a customer and an insurance agent. For example:
    
    {
    'label': ['claimStart', 'addDriver']
    'text': ['I was in a car accident and someone ran into my car but I don't know how to start the process.', 'My spouse just got their license and I want to ensure they're added to my policy.']
    }
    '''
USER_PROMPT = '''
    Provide 20 examples for each label, randomize the order of the examples.
    '''