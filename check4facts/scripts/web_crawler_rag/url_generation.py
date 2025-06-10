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
- Focus on **official, trustworthy, and domain-relevant sources only**. Choose sources depending on the topic of the claim.
- Avoid Greek blogs, news media, or journalist-run portals. Renowned global news agencies (e.g., Reuters, Associated Press) are acceptable.
- Include queries in **Greek and English**, if applicable.
- Use the `site:` operator to restrict search results to trustworthy domains.
- Make each query **highly specific** to the claim to locate information that confirms or denies it.


Dynamically adapt the sources based on the **claim’s domain**:

Some examples of domain-specific sources. Feel free to add more based on the claim's context, but avoid greek jounalistic sites or blogs:    

- **Health**: WHO, CDC, EMA, EODY, scientific journals (e.g., site:nature.com, site:thelancet.com)
- **Environment**: UNEP, IPCC, WWF, Greenpeace, site:ypen.gr
- **Migration**: UNHCR, IOM, Greek Ministry of Migration, Eurostat
- **Economy**: IMF, World Bank, ELSTAT (site:statistics.gr), OECD
- **Technology**: GSMA, IEEE, Gartner, IDC, ITU, Statista
- **Climate**: IPCC, WMO, NOAA, Copernicus, meteo.gr
- **Education**: Greek Ministry of Education, UNESCO, Eurydice, OECD
- **Agriculture**: FAO, Greek Ministry of Agriculture, Eurostat
- **Energy**: IEA, RAE.gr, ADMIE.gr, DEDDIE.gr, EU Energy

Instructions:
1. Identify the most relevant sources based on the claim’s domain.
2. Construct 10 focused Google search queries using both Greek and English where needed.
3. Use `site:` filters for official domains wherever possible.
4. Include the year (e.g., `2025`) or words like "recent", "latest", "updated", etc., to emphasize recency.
5. Do NOT include irrelevant government sites for global or unrelated topics.

Output Format:
Best 10 Google Search Queries WIHTOUT ANY ADDITIONAL TEXT OR COMMENTARY:
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
        print(f"LLM result is: {result}")
        queries = re.findall(r"\*+\s+(.*?)\s*$", result, re.MULTILINE)
        print(queries)
        return queries
