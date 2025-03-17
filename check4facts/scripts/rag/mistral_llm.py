import time
import numpy as np
import requests
import json
import os

class mistral_llm:
    def __init__(self, query, external_knowledge):

        self.external_knowledge = external_knowledge
        self.query = query
        
        self.prompt_without_rag = f'''
    
    You are given a statement: {query} that needs to be evaluated for accuracy.
        Use your knowledge to decide whether the statement is is ACCURATE, INACCURATE, RELATIVELY ACCURATE, or RELATIVELY INACCURATE.

        Before deciding:

        1. Clearly analyze the statement to understand its content and identify the key points 
        that need to be evaluated.
        2. Use your knowledge to assess the statement.

        Outcome: Provide a clear response by selecting one of the following labels:

        - ACCURATE: If the statement is fully confirmed by the information and evidence available to you.
        - INACCURATE: If the statement is clearly contradicted by the information and evidence available to you.
        - RELATIVELY ACCURATE: If the statement contains some correct elements but is not entirely accurate.
        - RELATIVELY INACCURATE: If the statement contains some correct elements but also includes misleading or inaccurate information.

        Finally, explain your reasoning clearly, focusing on the data provided and your own knowledge. 
        Avoid unnecessary details and strive to be precise and concise in your analysis. 
        Your responses should follow this format:
        Statement: 
        Statement Outcome: 
        Justification:

        Your answer should be in the Greek language, but the format in English. Do not use the (*) symbol.

    '''

        self.prompt_with_rag = f'''
    You have at your disposal information '[Information]' that was found on the web, and a statement: '[User Input]' whose accuracy must be evaluated. 
    Use only the provided information from the web in combination with your knowledge to decide whether the statement is is ACCURATE, INACCURATE, RELATIVELY ACCURATE, or RELATIVELY INACCURATE.

    Before you decide:

    1. Analyze the statement clearly to understand its content and identify the main points that need to be evaluated.
    2. Compare the statement with the information you have, evaluating each element of the statement separately.
    3. Use your knowledge ONLY in combination with the provided information, avoiding reference to unverified information.

    Result: Provide a clear answer by choosing one of the following labels:

    - ACCURATE: If the statement is fully confirmed by the information and evidence you have.
    - INACCURATE: If the statement is clearly disproved by the information and evidence you have.
    - RELATIVELY ACCURATE: If the statement contains some correct elements, but not entirely accurate.
    - RELATIVELY INACCURATE: If the statement contains some correct elements but also contains misleading or inaccurate information.

    The statement and the external knowledge are listed below:

    Finally, explain your reasoning clearly and focus on the provided data and your own knowledge. Avoid unnecessary details and try to be precise and concise in your analysis. Your answers should be in the following format:

    Statement: 
    Result of the statement:
    Justification:

    statement: {query}
    external knowledge: {external_knowledge}

    Your answer should be in the Greek, but the format in English. Do not use the (*) symbol. 
    The justification should be a paragraph long, where you explain you reasoning.

    '''
        

   
    def run_mistral_llm(self, article_id):
        print('Invoking mistral llm...')
        try:
            article_id = int(article_id)
        except ValueError:
            print("Error: article_id is not an integer")
            return None


        url = "https://api.mistral.ai/v1/chat/completions"
        

        if  self.external_knowledge is not None:
            system_content = self.prompt_with_rag
            content = f"Query: {self.query}\nInformation: {str(self.external_knowledge)}"
        else:
            system_content = self.prompt_without_rag
            content = f"Query: {self.query}"
        # Prepare the payload
        payload = json.dumps({
            "model": "mistral-large-latest", 
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": content}],
            "temperature": 0    
        })
        # Set the request headers
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {os.getenv("MISTRAL_API_KEY")}"
        }
        
        # Send the request
        response = requests.post(url, headers=headers, data=payload)
        # Check for success
        if response.status_code == 200:
            response_json = response.json()
            ai_message = response_json['choices'][0]['message']['content']
            print({"response": ai_message,
                     "article_id": article_id,  "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())})
            return {"response": ai_message,
                     "article_id": article_id,  "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
