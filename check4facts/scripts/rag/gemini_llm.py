
import time
import numpy as np

class gemini_llm:
    def __init__(self, query, external_knowledge):

        self.external_knowledge = external_knowledge

        
        self.prompt_without_rag = f'''
    
    You are given a statement: {query} that needs to be evaluated for accuracy.
        Use your knowledge to decide whether the statement is TRUE, FALSE, PARTIALLY-TRUE, or PARTIALLY-FALSE.

        Before deciding:

        1. Clearly analyze the statement to understand its content and identify the key points 
        that need to be evaluated.
        2. Use your knowledge to assess the statement.

        Outcome: Provide a clear response by selecting one of the following labels:

        - TRUE: If the statement is fully confirmed by the information and evidence available to you.
        - FALSE: If the statement is clearly contradicted by the information and evidence available to you.
        - PARTIALLY-TRUE: If the statement contains some correct elements but is not entirely accurate.
        - PARTIALLY-FALSE: If the statement contains some correct elements but also includes misleading or inaccurate information.

        Finally, explain your reasoning clearly, focusing on the data provided and your own knowledge. 
        Avoid unnecessary details and strive to be precise and concise in your analysis. 
        Your responses should follow this format:
        Statement: 
        Statement Outcome: 
        Justification:

        Your answer should be in the Greek language.

    '''

        self.prompt_with_rag = f'''
    You have at your disposal information '[Information]' and a statement: '[User Input]' whose accuracy must be evaluated. 
    Use only the provided information in combination with your knowledge to decide whether the statement is TRUE, FALSE, PARTIALLY TRUE, or PARTIALLY FALSE.

    Before you decide:

    1. Analyze the statement clearly to understand its content and identify the main points that need to be evaluated.
    2. Compare the statement with the information you have, evaluating each element of the statement separately.
    3. Use your knowledge ONLY in combination with the provided information, avoiding reference to unverified information.

    Result: Provide a clear answer by choosing one of the following labels:

    - TRUE: If the statement is fully confirmed by the information and evidence you have.
    - FALSE: If the statement is clearly disproved by the information and evidence you have.
    - PARTIALLY TRUE: If the statement contains some correct elements, but not entirely accurate.
    - PARTIALLY FALSE: If the statement contains some correct elements but also contains misleading or inaccurate information.

    The statement and the external knowledge are listed below:

    Finally, explain your reasoning clearly and focus on the provided data and your own knowledge. Avoid unnecessary details and try to be precise and concise in your analysis. Your answers should be in the following format:

    Statement: 
    Result of the statement:
    Justification:

    statement: {query}
    external knowledge: {external_knowledge}

    Your answer should be in the Greek. 

    '''











    def google_llm(self, article_id):
        print('Invoking gemini llm...')
        import google.generativeai as genai
        import os
        from dotenv import load_dotenv
        import logging
        os.environ["GRPC_VERBOSITY"] = "none"
        logging.getLogger("absl").setLevel(logging.CRITICAL)
        logging.basicConfig(level=logging.ERROR)
        load_dotenv()


        try:
            article_id = int(article_id)
        except ValueError:
            print("Error: article_id is not an integer")
            return None

         

        try:
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            model = genai.GenerativeModel("gemini-1.5-flash")
            if self.external_knowledge:
                response = model.generate_content(self.prompt_with_rag)
            else: 
                response = model.generate_content(self.prompt_without_rag)
                
        

            
                
            return {"response": response.text,
                     "article_id": article_id,  "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
        except Exception as e:
                print(f"Error occured during the Gemini model invokation: {e}")
                return None