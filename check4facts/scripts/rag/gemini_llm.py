import time
import numpy as np


class gemini_llm:
    def __init__(self, query, external_knowledge):

        self.external_knowledge = external_knowledge

        self.prompt_without_rag = f"""
    
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
        - UNVERIFIABLE: If you cannot verify the accuracy of the claim.

        Finally, explain your reasoning clearly, focusing on the data provided and your own knowledge. 
        Avoid unnecessary details and strive to be precise and concise in your analysis. 
        Your responses should follow this format:
        Statement: 
        Result of the statement: 
        Justification:

        Your answer should be in the Greek language. Do not use the (*) symbol. 

    """

        self.prompt_with_rag = f"""
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
    - UNVERIFIABLE: If you cannot verify the accuracy of the claim based on the information.

    The statement and the external knowledge are listed below:

    Finally, explain your reasoning clearly and focus on the provided data and your own knowledge. Avoid unnecessary details and try to be precise and concise in your analysis. Your answers should be in the following format:

    Statement: 
    Result of the statement:
    Justification:

    statement: {query}
    external knowledge: {external_knowledge}

    Your answer should be in the Greek. Do not use the (*) symbol. 
    Do NOT mention specific sources, documents, or line numbers. 
    Simply provide a well-formed justification in your own words, in a paragraph. 
    The justification should be a paragraph long, where you explain you reasoning.


    """

    def google_llm(self, article_id):
        print("Invoking gemini llm...")
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
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel("gemini-2.0-flash")
            if self.external_knowledge is not None:
                response = model.generate_content(self.prompt_with_rag)
            else:
                response = model.generate_content(self.prompt_without_rag)

            return {
                "response": response.text,
                "article_id": article_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            }
        except Exception as e:
            print(f"Error occured during the Gemini model invokation: {e}")
            return None
