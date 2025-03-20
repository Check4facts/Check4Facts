import os
from langchain_groq import ChatGroq
import time
import numpy as np
from check4facts.scripts.rag.translate import *


class groq_api:

    def __init__(self, info, query):

        self.info = info
        self.query = query

        self.llm = ChatGroq(
            model=os.getenv(
                "GROQ_LLM_MODEL_1"
            ),  # Note: changed the LLM invoke order for debugging
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=1,
            api_key=os.getenv("GROQ_API_KEY_1"),
        )

        self.llm_2 = ChatGroq(
            model=os.getenv("GROQ_LLM_MODEL_2"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=1,
            api_key=os.getenv("GROQ_API_KEY_2"),
        )

    def run_api(self, article_id):
        max_retries = 5
        retries = 0

        try:
            article_id = int(article_id)
        except ValueError:
            print("Error: article_id is not an integer")
            return None

        if self.info is not None:
            messages = [
                (
                    "system",
                    f"""You have at your disposal information '[Information]' that was found on the web, and a statement: '[User Input]' whose accuracy must be evaluated. 
Use only the provided information from the web, in combination with your knowledge to decide whether the statement is ACCURATE, INACCURATE, RELATIVELY ACCURATE, or RELATIVELY INACCURATE.

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

Finally, explain your reasoning clearly and focus on the provided data and your own knowledge. Avoid unnecessary details and try to be precise and concise in your analysis. Your answers should be in the following format:

Statement: '[User Input]'
Result of the statement:
Justification:

Your answer should be in Greek language. The format should be in English.
""",
                ),
                (
                    "human",
                    f'''External info '{self.info}'
                         Statement: '{self.query}' "''',
                ),
            ]

        else:
            messages = [
                (
                    "system",
                    f"""
You are given a statement: '[statement]' that needs to be evaluated for accuracy.
Use your knowledge to decide whether the statement is ACCURATE, INACCURATE, RELATIVELY ACCURATE, or RELATIVELY INACCURATE.

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
Statement: '[statement under examination]'
Statement Outcome: 
Justification:

Your answer should be in Greek language. The format should be in English.
""",
                ),
                (
                    "human",
                    f""" Analyse the following statement: {self.query}
        """,
                ),
            ]
        ai_msg = None

        while retries < max_retries:
            time.sleep(5)

            # try to create an api call with the first llm
            try:
                ai_msg = self.llm.invoke(messages)
                break

            # if it fails to produce a result, try with the second llm
            except Exception as e:
                try:
                    ai_msg = self.llm_2.invoke(messages)
                    break

                # if a second llm doesn't work either, try another time
                except Exception as e:
                    retries += 1
                    time.sleep(5)

        # if no answer could be generated, return none and invoke the local llm
        if ai_msg is None:
            return {"response": None}

        return {
            "response": ai_msg.content,
            "article_id": article_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
