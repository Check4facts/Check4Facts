import os
os.environ["GROQ_API_KEY"] = '<API_KEY_1>' 
from langchain_groq import ChatGroq
import time
import numpy as np


class groq_api():

    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=1,)
        
        self.llm_2 = ChatGroq(
            model="llama3-70b-8192",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=1,)
        
        self.key_1 = '<API_KEY_1>'
        self.key_2 = '<API_KEY_2>'

  
        
    def run(self, text):
        max_retries = 10
        retries= 0
        messages = [
                    (
                        "system",
                        f"""You are a text summarizer. Summarize the following text in the form of a short bulleted list.
                        Keep the sentences and the list short and to the point. Do not make any commentary, just provide the summary.
                        """,

                        #On the LAST BULLET, infer the validity of the claim I provide below, based on the text that you read.
                        #Claim: {claim}
                    ),
                    ("human", f"{text}"),
                ]
        ai_msg = None

        while retries<max_retries:

            #alternate between api keys
            if retries % 2 ==0:
                 os.environ["GROQ_API_KEY"] = self.key_1
            else: 
                 os.environ["GROQ_API_KEY"] = self.key_2

            #try to create an api call with the first llm     
            try:
                #print()
                #print(f'Invoking with model: {self.llm.model_name}')
                #print(f'And key: {str(os.environ["GROQ_API_KEY"])[:3]}....{str(os.environ["GROQ_API_KEY"])[-3:]}')
                #print()
                start_time = time.time()
                ai_msg = self.llm.invoke(messages)
                #print(ai_msg.content)
                #print()
                end_time = time.time()
                break

            #if it fails to produce a result, try with the second llm
            except Exception as e:
                try:
                    #print()
                    #print(f'Invoking with model: {self.llm_2.model_name}')
                    #print(f'And key: {str(os.environ["GROQ_API_KEY"])[:3]}.......{str(os.environ["GROQ_API_KEY"])[-3:]}')
                    #print()
                    start_time = time.time()
                    ai_msg = self.llm_2.invoke(messages)
                    #print(ai_msg.content)
                    #print()
                    end_time = time.time()
                    break

                #if a second llm doesn't work either, alternate between api keys by increasing the #retries                     
                except Exception as e:
                    #print(e)
                    retries+=1

        #if no answer could be generated, return none and invoke the local llm
        if ai_msg is None:  
            #print("All retries failed. Unable to invoke LLM.")
            #print('Initializing localhost llm...')
            return {"response" : None, "elapsed_time": None}
      
        return {"response" : ai_msg.content,
                        "elapsed_time": np.round(end_time-start_time,2)}
            