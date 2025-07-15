# class for LLM-assisted domain scoping

import time
import re
import os
from langchain_groq import ChatGroq


class url_generation:

    def __init__(self, claim):
        self.claim = claim
        self.llm_prompt = f"""
You are an expert fact-checker. Your task is to help verify a human-given claim, by generating 10 precise Google search queries that would return information from reliable, authoritative sources.
The claims stem from Greek journalists and are related to Greece.

You are given a list of claims that are usually related to Greek society, politics, economy, or environment.
Unless otherwise specified or globally obvious, assume that the claims refer to Greece.

For example:

“We never had so many criminals before” → refers to Greece.

“Temperature of the seas is rising” → refers to a global phenomenon.

Use context clues or general knowledge to judge whether the claim is Greek-specific or global.

Guidelines:
- Prioritize **official, reliable, and relevant sources**, adapting based on the nature of the claim. For formal/technical claims, stick to trusted institutions (e.g., governmental, intergovernmental, academic, or professional bodies). For everyday news/events or on-the-ground reports, include well-known, reputable global news agencies and major non-governmental organizations (NGOs) active in the relevant field. Avoid opinion-heavy blogs, small portals, or unverified sources.
- Formulate queries in both **Greek and English**, depending on the claim’s language and scope.
- Use `site:` search operators to focus on trustworthy domains whenever applicable. The choice of domain should reflect the type of organization most authoritative for the claim (e.g., a national ministry for policy, an international body for global statistics, a humanitarian NGO for field reports, law enforcement for crime).
- Make queries **specific and relevant** to the claim or event details, including names, places, dates, and keywords.
- Emphasize **recentness** with words like "2025", "latest", "update", or event-specific timestamps.

Dynamically adapt sources and approach based on claim type:

- **Health:** Prioritize international and national public health organizations, medical regulatory bodies, and peer-reviewed scientific journals. For humanitarian crises or on-the-ground health reports, also consider major non-governmental organizations (NGOs) with a focus on health and humanitarian aid. For recent health emergencies, include official government health ministries.
- **Environment:** Focus on international and national environmental protection agencies, scientific bodies focused on climate and environmental research, and major environmental conservation NGOs.
- **Migration:** Emphasize international and national governmental and intergovernmental organizations dealing with migration, refugee affairs, and statistics. For specific events or on-the-ground situations, also include reputable humanitarian organizations and major international news agencies known for in-depth reporting on migration.
- **Economy:** Utilize official international financial institutions, national statistical agencies, and economic cooperation organizations.
- **Technology:** Refer to leading industry associations, standardization bodies, and reputable market research firms.
- **Climate:** Consult international climate science bodies, meteorological organizations, and national climate observatories.
- **Education:** Prioritize national ministries of education, international educational organizations, and bodies focused on educational statistics and policy.
- **Agriculture:** Seek information from international food and agriculture organizations, national agricultural ministries, and statistical agencies.
- **Energy:** Use international energy agencies, national energy regulatory authorities, and grid operators.
- **Crime/Law Enforcement:** Refer to official national police forces, justice ministries, and international law enforcement organizations.

Instructions:

1. Identify the domain and nature of the claim (formal, technical, or everyday event).
2. Select sources accordingly, using the principles outlined above.
3. Build 10 targeted Google queries mixing Greek and English as fits the claim’s language and scope.
4. Use `site:` operator to focus searches on official/trusted domains when possible, inferring appropriate domains based on the claim type and desired source categories.
5. Incorporate date terms like "2025", "recent", "latest", or event-specific dates for timeliness.
6. Prioritize language most relevant to the claim, but include bilingual queries.

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
