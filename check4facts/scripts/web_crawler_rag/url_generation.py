# class for LLM-assisted domain scoping

import time
import re
import os
from langchain_groq import ChatGroq


class url_generation:

    def __init__(self, claim):
        self.claim = claim
        self.llm_prompt = (
            prompt
        ) = f"""
You are an expert fact-checker. Your task is to help verify the following claim by generating 10 precise Google search queries that would return information from reliable, authoritative sources.

Claim:
{claim}

Guidelines:
- Focus on official and trustworthy sources only:
  - Greek authorities
  - International authorities
  - Scientific journals or organizations that could help with the claim's domain
  -
- Do NOT include blogs, news media, or journalist-run portals.
- Include queries in Greek **and** English if relevant.
- Use the `site:` operator to restrict search results to specific domains.
- Make each query highly specific to help locate information that confirms or denies the claim.
- Start with the most relevant query.
- Change the URLs provided depending on the claim's domain. For example, 
  if the claim is enviroment related, search for environmental organizations, if the claim is migrations related, 
  check about migration organizations and authorities etc.
- Focus on **fresh and recent information**, ideally published in **the last 1–2 years**. You can encourage this by:
  - Including the year (e.g., `2025`) in the query
  - Adding terms like `recent`, `new`
- Make each query highly specific to help locate information that confirms or denies the claim.
Do not make any commentary, do not provide explanations. Just provide the queries in the format below:

Output Format:
Top 10 Google Search Queries:
* ...
* ...
...
* ...
"""

        self.llm = ChatGroq(
            model=os.getenv("GROQ_LLM_MODEL_1"),
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

    def google_llm(self):
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
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel("gemini-2.0-flash")

            response = model.generate_content(self.llm_prompt)

            return response.text

        except Exception as e:
            print(f"Error occured during the Gemini model invokation: {e}")
            return None

    def run_groq(self):
        max_retries = 5
        retries = 0

        messages = [
            (
                "system",
                self.llm_prompt,
            ),
            (
                "human",
                f" Claim: {self.claim}",
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
            return None

        return ai_msg.content

    def generate_queries(self):

        # result = self.google_llm()
        result = self.run_groq()
        # print(f"LLM result is: {result}")
        urls = re.findall(r"\*+\s+(.*?)\s*$", result, re.MULTILINE)
        print(urls)
        return urls
