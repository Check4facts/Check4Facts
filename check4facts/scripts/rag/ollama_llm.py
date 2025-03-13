
import time
import numpy as np
import ollama
import re
from check4facts.scripts.text_sum.translate import translate, translate_long_text

class ollama_llm:
    def __init__(self, query, external_knowledge):
        self.external_knowledge = external_knowledge
        self.query = query
        self.model = "mistral:instruct"
        self.prompt_without_rag = f'''
    
    You are given a statement: {self.query} that needs to be evaluated for accuracy.
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
        Your responses should follow this format in English only:
        Statement: 
        Statement Outcome: 
        Justification:

       

    '''

        self.prompt_with_rag = f'''
    You have at your disposal information '[Information]' and a statement: '[User Input]' whose accuracy must be evaluated. 
    Use only the provided information in combination with your knowledge to decide whether the statement is is ACCURATE, INACCURATE, RELATIVELY ACCURATE, or RELATIVELY INACCURATE.

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

    statement: {self.query}
    external knowledge: {self.external_knowledge}

    

    '''
    
    #method to remove 20% of text for cases when the input token limit is exceeded 
    def truncate_text(self, text):
        words = text.split()
        total_words = len(words)
        words_to_keep = int(total_words * 0.8)
        truncated_text = ' '.join(words[:words_to_keep])
        sentence_end = re.search(r'\.|\?|!$', truncated_text)  
        if sentence_end:
            truncated_text = truncated_text[:sentence_end.end()]
        
        return truncated_text

    def run_ollama_llm(self, article_id):
        retries = 0
        try:
            article_id = int(article_id)
        except ValueError:
            print("Error: article_id is not an integer")
            return None


        print('Invoking local llm...')
        if(self.external_knowledge is not None):
            print('------------------External knowledge----------------------')
            print(self.external_knowledge)
            print('----------------------------------------------------------')
        while retries<=3:
            try:
                if retries>=3:
                    print(f'Error: Could not generate output for model: {self.model}')
                if self.external_knowledge is not None:
                    response = ollama.chat(model=self.model, messages=[{"role": "user", "content": self.prompt_with_rag}])
                break
            except Exception as e:
                print(e)
                print("Retrying with smaller input....")
                self.external_knowledge = self.truncate_text(self.external_knowledge)
                retries+=1
                
        
        answer = response['message']['content']
        #answer = translate_long_text(answer, src_lang='en', target_lang='el')
       

        # Output the response from the model
        return {"response": answer,
                "article_id": article_id,  "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
