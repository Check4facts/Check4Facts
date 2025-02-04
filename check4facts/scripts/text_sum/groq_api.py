import os
from langchain_groq import ChatGroq
import time
import numpy as np
from check4facts.scripts.text_sum.translate import translate_long_text
from check4facts.scripts.text_sum.text_process import *



class groq_api:

    def __init__(self):

        self.llm = ChatGroq( 
            model=os.getenv("GROQ_LLM_MODEL_1"), 
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=1,
            groq_api_key=os.getenv("GROQ_API_KEY_1"),
        )

        self.llm_2 = ChatGroq(
            model=os.getenv("GROQ_LLM_MODEL_2"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=1,
            groq_api_key=os.getenv("GROQ_API_KEY_2"),
        )



    def run(self, text):
        max_retries = 10
        retries = 0
        messages = [
            (
                "system",
                f"""You are a text summarizer. Summarize the following text in the form of a short bulleted list, meaning 3 or 4 bullets.
                        Keep the sentences and the list short and to the point. Do not make any intoduction, just provide the summary.
                        DO NOT WRITE "HERE IS A SUMMARY" OR SOMEHTING RELEVANT.
                        """,
                
            ),
            ("human", f"{text}"),
        ]
        ai_msg = None

        while retries < max_retries:

            # try to create an api call with the first llm
            try:
                start_time = time.time()

                if len(text.split())>=1800:
                    print('Summarizing long text with the first LLM')
                    time.sleep(5)
                    ai_msg = self.summarize_long_text(self.llm, text)
                    return {
                            "response": ai_msg,
                            "elapsed_time": np.round(time.time() - start_time, 2),
                        }
                else:
                    time.sleep(5)
                    ai_msg = self.llm.invoke(messages)
                    
                    
                break

            # if it fails to produce a result, try with the second llm
            except Exception as e:
                print(f"Error: {e}")
                print('Failed to produce a result, try with the second llm')
                try:
                     start_time = time.time()
                     time.sleep(5)
                     if len(text.split())>=1800:
                         print('Summarizing text with the second LLM')
                         ai_msg = self.summarize_long_text(self.llm_2, text)
                         return {
                            "response": ai_msg,
                            "elapsed_time": np.round(time.time() - start_time, 2),
                        }
                     else:
                         time.sleep(5)
                         ai_msg = self.llm_2.invoke(messages)
                     
                     break

                # if a second llm doesn't work either, try increasing the #retries and wait 
                except Exception as e:
                    print(f'Exception: {e}')
                    retries += 1
                    time.sleep(5)

        # if no answer could be generated, return none and invoke the local llm
        if ai_msg is None:
            print('Failed to produce result. Trying with the gemini llm next....')
            return None
        
        text = translate_long_text(ai_msg.content, src_lang='en', target_lang='el')
        text = text_to_bullet_list(text)
        text = bullet_to_html_list(text)

        return {
            "response": text,
            "elapsed_time": np.round(time.time() - start_time, 2),
        }
    

    def summarize_long_text(self, llm, text):
        start_time = time.time()
        from langchain.prompts import PromptTemplate
        from langchain.chains.summarize import load_summarize_chain
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        llm_name = llm.model_name

        #Dynamically split the text into three parts
        chunk_size = len(text) // 3  
        chunk_1 = text[:chunk_size]
        chunk_2 = text[chunk_size:2*chunk_size]
        chunk_3 = text[2*chunk_size:]  


        

        # Define the map template
        map_template = """Summarize the provided text.
                      Keep the summary short and concise. Do not make any commentary, just provide the summary.
                      Text to be summarized: {docs}"""
        map_prompt = PromptTemplate.from_template(map_template)

       # Define the reduce template
        reduce_template = """Summarize the following text in the form of a short bulleted list. Generate 5 bullets at most. 
                         Keep the sentences and the list short and to the point. Do not make any commentary, just provide the summary.
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
        
        result_1_text = result_1.content if hasattr(result_1, 'content') else str(result_1)
        result_2_text = result_2.content if hasattr(result_2, 'content') else str(result_2)
        result_3_text = result_3.content if hasattr(result_3, 'content') else str(result_3)

        # Use the reduce_chain to generate the final summary from the combined results
        
        combined_results = "\n".join([result_1_text, result_2_text, result_3_text])
       
        final_summary = reduce_chain.invoke({"docs": combined_results})

        print('FINAL SUMMARY: ')
        print(final_summary.content if hasattr(final_summary, 'content') else str(final_summary))
        result = final_summary.content if hasattr(final_summary, 'content') else str(final_summary)

        text = translate_long_text(result, src_lang='en', target_lang='el')
        text = text_to_bullet_list(text)
        text = bullet_to_html_list(text)
        

        return {
            "response": text,
            "elapsed_time": np.round(time.time() - start_time, 2),
        }



        # # Initialize text splitter
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
        # docs = text_splitter.create_documents([text])
        # print(f"Created {len(docs)} documents")
        
        # # Define the map template
        # map_template = """Summarize the following text in the form of a short bulleted list.
        #                 Keep the sentences and the list short and to the point. Do not make any commentary, just provide the summary.
        #                 Text to be summarized: {docs}"""
        
        # map_prompt = PromptTemplate.from_template(map_template)
        # # Define the reduce template
        # reduce_template = """The following is a set of summaries:
        #                     {docs}
        #                     Summarize the following text in the form of a short bulleted list. 
        #                     Keep the sentences and the list short and to the point. Do not make any commentary, just provide the summary."""
        
        # reduce_prompt = PromptTemplate.from_template(reduce_template)
        # # Use the new syntax for creating the map and reduce steps
        # map_chain = map_prompt | llm  # Chain map prompt with LLM
        # reduce_chain = reduce_prompt | llm  # Chain reduce prompt with LLM
        # # Create the map-reduce chain using RunnableMapReduce
        
        # map_reduce_chain = load_summarize_chain(
        #     llm=ChatGroq(model=llm_name),
        #     chain_type="map_reduce",
        #     return_intermediate_steps=False
        # )
        # # Run the chain
        
        # final_summary = map_reduce_chain({"input_documents": docs})
        # print('FINAL SUMMARY: ')
        # print(final_summary)
        # return final_summary