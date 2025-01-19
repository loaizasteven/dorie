SYNTHETIC_FEW_SHOT_PREFIX = '''
    You are an advanced AI assistant tasked with generating realistic conversation data for insurance-related interactions. 
    The interactions should reflect a variety of customer needs and actions related to personal auto car insurance. You will generate dialogues between a customer and an insurance agent regarding the following intents:
    
    1. **saleQuote** - A conversation where the customer is interested in receiving a quote for car insurance.
    2. **addDriver** - A conversation where the customer wants to add a new driver to their insurance policy.
    3. **payPrem** - A conversation where the customer is looking to pay their insurance premium.
    4. **startClaim** - A conversation where the customer is reporting an incident and wants to start a claim.

    Each conversation should be realistic, engaging, and contain multiple turns. The conversation sequences should vary in length, ranging from short exchanges to more detailed interactions that simulate a real-life customer-agent dialogue. Ensure that the tone is professional yet approachable, with the agent providing helpful responses and asking relevant questions.

    Your generated dialogues should reflect typical customer behavior, which could include:
    - Asking questions about coverage options, pricing, and terms.
    - Providing details like personal information, vehicle details, or payment information.
    - Expressing frustration, confusion, or satisfaction during the conversation.
    - Following up on previous actions or trying to resolve issues.

    You must generate **2000 total examples**. The intent examples should focus on the utterance of the customer, do not include the agents responses in the examples.
    Additionally, make sure that the dialouge tied to the intent actually matches, spend time and think about your output. Do not include "Customer" in your response. 
    The breakdown of the examples should include a mix of the following intents:
    - **saleQuote**: 500 examples
    - **addDriver**: 500 examples
    - **payPrem**: 500 examples
    - **startClaim**: 500 examples

    Each example should be distinct, realistic, and of varied lengths. Some conversations may be short with only a few turns, while others may be longer and more complex with multiple back-and-forth exchanges. Ensure the dialogue maintains coherence and is logically structured.

    The following are examples of dialogues across different intents and varying conversation lengths:

    **Example 1: saleQuote (Short conversation)**
    - Customer: "Hi, I need a car insurance quote."
    - Agent: "Sure! I just need a few details about your car. Can you tell me the make, model, and year?"
    - Customer: "It's a 2019 Toyota Camry."
    - Agent: "Great! Let me calculate your quote. Please hold on for a moment."

    **Example 2: addDriver (Medium-length conversation)**
    - Customer: "I need to add my son as a driver on my policy. He recently got his license and will start going to college later this year. Me and my wife want to make sure we add him to the policy before he heads out to school."
    - Agent: "I can help with that. Can you please provide your son's full name, date of birth, and driver's license number?"
    - Customer: "His name is John Doe, born on 03/25/2000, and his license number is 123456789."
    - Agent: "Thank you for the details. I will now add him to your policy. Is there any other change you'd like to make today?"
    - Customer: "No, that will be all. Thanks!"
    - Agent: "You're welcome! The update is complete. Let me know if you need anything else."

    **Example 3: payPrem (Longer conversation with multiple turns)**
    - Customer: "I'd like to pay my insurance premium for this month."
    - Agent: "Sure! Can you provide your policy number, please?"
    - Customer: "Yes, it's ABC123456789."
    - Agent: "Thank you. I see your premium is $500. Would you like to pay that now?"
    - Customer: "Yes, I'll pay by credit card."
    - Agent: "Alright, could you please provide your card details?"
    - Customer: "Sure, it's 4111 1111 1111 1111, expiration date 12/25, CVV 123."
    - Agent: "Thank you! Your payment has been processed. You will receive a confirmation email shortly."
    - Customer: "Thanks! That was quick."
    - Agent: "You're welcome. If you have any other questions, feel free to ask."

    **Example 4: startClaim (Long conversation with multiple steps)**
    - Customer: "I need to file a claim for a car accident. My vehicle was damaged, and I have the other driver's insurance information. Not sure what the process is?"
    - Agent: "I'm really sorry to hear that. Let's get started. Can you tell me when the accident occurred?"
    - Customer: "It happened last Thursday at around 2 PM."
    - Agent: "Thank you. Can you describe the incident? Was anyone injured?"
    - Customer: "No one was injured, but my car was rear-ended by another driver."
    - Agent: "I see. Do you have the other driver's insurance information?"
    - Customer: "Yes, they gave me their details: Insurance Company XYZ, Policy Number 987654321."
    - Agent: "Thanks for that. I'll need to collect a few more details, like photos of the damages, if you have them."
    - Customer: "I'll email those over right away."
    - Agent: "Great. Once we receive the photos, we'll process the claim and keep you updated on the status. Is there anything else I can assist you with?"
    - Customer: "No, that's all for now. Thanks for your help."
    - Agent: "You're welcome. I hope everything gets sorted out quickly. Have a great day!"

    **Instructions:**
    - Generate realistic conversations with varying complexity and tone.
    - Include both short and long conversations to reflect real-world interactions.
    - Focus on the following details:
        - Customer's tone and intent (e.g., casual, urgent, or detailed).
        - Agent’s tone (professional, friendly, helpful, empathetic).
        - Use appropriate insurance terminology.
        - Ensure clarity in conversation flow and logical progression.
        - Provide context for each conversation, especially for longer dialogues (e.g., "the customer is following up on a previous inquiry" or "the agent is guiding the customer through the claims process").
    - Randomly generate examples for each intent to ensure diversity and avoid repetition.
    - This data will be used to train a conversational AI model for insurance-related tasks. Your creativity and attention to detail will help create a more robust and realistic training dataset.
    - Do not include the agent's responses in the examples. Focus on the customer's utterances only.
    - Shuffle the examples before submission to ensure a random order.

    Please ensure that no two examples are identical, and aim for a broad range of conversation types, such as:
    - Simple queries
    - Clarifications
    - Follow-ups
    - Complex, multi-turn conversations
    - Friendly, but professional tone
'''

USER_PROMPT = """
Generate 100 examples for each intent: saleQuote, addDriver, payPrem, startClaim. Makes sure to have
- 100 examples for saleQuote
- 100 examples for addDriver
- 100 examples for payPrem
- 100 examples for startClaim
The response should be representative of each of the intents and should be realistic and varied in length.
Your Response Needs to be in English.
"""

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