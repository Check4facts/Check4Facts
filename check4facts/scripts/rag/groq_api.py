import os
from langchain_groq import ChatGroq
import time
import numpy as np
import re
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

            # If external information harvested is longer than 1000 words, summarize them to fit the llm
            if len(str(self.info).split(" ")) >= 800:
                try:
                    print("Input text is too long. Summarizing external info....")
                    self.info = self.summarize_long_text(self.llm, self.info)
                    print(self.info)
                except Exception as e:
                    print(f"Error during summarixation: {e}")
                    print("Trying with the another model....")
                    self.info = self.summarize_long_text(self.llm_2, self.info)

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

    def summarize_long_text(self, llm, text):
        start_time = time.time()
        from langchain.prompts import PromptTemplate

        # Dynamically split the text into three parts
        chunk_size = len(text) // 3
        chunk_1 = text[:chunk_size]
        chunk_2 = text[chunk_size : 2 * chunk_size]
        chunk_3 = text[2 * chunk_size :]

        # Define the map template
        map_template = """Summarize the provided text in Greek language.
                      Keep the important points of the text. Do not write "HERE IS A SUMMARY" or something relevant.
                      Text to be summarized: {docs}"""
        map_prompt = PromptTemplate.from_template(map_template)

        # Define the reduce template
        reduce_template = """Summarize the following text. Keep the important parts but keep the content also rich.
        Avoid rambling or unnecessary details. 
          Do not write "HERE IS A SUMMARY" or something relevant.
                         Answer in Greek language.
                         Text to summarize: {docs}"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)

        # Use the new syntax for creating the map and reduce steps
        map_chain = map_prompt | llm  # Chain map prompt with LLM
        reduce_chain = reduce_prompt | llm  # Chain reduce prompt with LLM

        # Process each chunk with the map_chain (i.e., invoke the LLM three times)
        result_1 = map_chain.invoke({"docs": chunk_1})
        time.sleep(5)
        result_2 = map_chain.invoke({"docs": chunk_2})
        time.sleep(5)
        result_3 = map_chain.invoke({"docs": chunk_3})
        time.sleep(5)

        # Combine the results into one final summary

        result_1_text = (
            result_1.content if hasattr(result_1, "content") else str(result_1)
        )
        result_2_text = (
            result_2.content if hasattr(result_2, "content") else str(result_2)
        )
        result_3_text = (
            result_3.content if hasattr(result_3, "content") else str(result_3)
        )

        # Use the reduce_chain to generate the final summary from the combined results

        combined_results = "\n".join([result_1_text, result_2_text, result_3_text])

        final_summary = reduce_chain.invoke({"docs": combined_results})

        print("FINAL SUMMARY: ")
        # print(final_summary.content if hasattr(final_summary, 'content') else str(final_summary))
        result = (
            final_summary.content
            if hasattr(final_summary, "content")
            else str(final_summary)
        )

        return result
