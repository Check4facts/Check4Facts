from check4facts.scripts.text_sum.translate import *
import numpy as np
from check4facts.scripts.text_sum.text_process import *

def initialize_model(max_length):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    import torch
    
    model_name = "google-t5/t5-small"
    device = 0 if torch.cuda.is_available() else "cpu"  # Use GPU if available, else use CPU
    
    # Load model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16  
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 1024*5  # Set the max length of the input text
    model.to(device)
    
    # Return summarization pipeline
    return pipeline("summarization", model=model, tokenizer=tokenizer, device=device, 
                    max_length=max_length, truncation=True)



def generate_summary(text):
    input_length = len(text.split())
    max_length = min(1024*2, max(20, int(input_length * 0.5)))  
    summarizer = initialize_model(max_length)
    summary = summarizer(text, num_return_sequences=1)
    return summary[0]['summary_text']

def invoke_local_llm(text, article_id):

    try:
        article_id = int(article_id)
    except ValueError:
        print("Error: article_id is not an integer")
        return {"error": "Invalid article_id"} 

    start_time = time.time()
    print('Invoking local llm...')
    text = translate_long_text(text, src_lang='el', target_lang='en')
    summary = generate_summary(text)
   
    end_time = time.time()
    result_sum = translate_long_text(summary,src_lang='en', target_lang='el')
    elapsed_time = np.round(end_time-start_time,2)

    bulleted_list = text_to_bullet_list(result_sum)

    print(bulleted_list)
    
    bulleted_list = bullet_to_html_list(bulleted_list)

    print(bulleted_list)

    bulleted_list = capitalize_bullets(bulleted_list)

    print(bulleted_list)

    return {"summarization": bulleted_list, "time": elapsed_time, "article_id": article_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) }



def google_llm(text, article_id):
    import google.generativeai as genai
    import os
    from dotenv import load_dotenv
    import logging
    os.environ["GRPC_VERBOSITY"] = "none"
    logging.getLogger("absl").setLevel(logging.CRITICAL)
    logging.basicConfig(level=logging.ERROR)
    load_dotenv()
    api_key = os.getenv("API_KEY")

    try:
        article_id = int(article_id)
    except ValueError:
        print("Error: article_id is not an integer")
        return None

    if not api_key:
        print("Error: article_id is not an integer")
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        start_time = time.time()
        text = translate_long_text(text, src_lang='el', target_lang='en')
        response = model.generate_content(f'''In a short dashed list, summarize the following text. 
                                        Do not make any commentary. Just provide the summary. Use at most 5 dashes.
                                        The text to be summarized is presented below:
                                        {text}''')
        
        text = translate_long_text(response.text, src_lang='en', target_lang='el')
        text = bullet_to_html_list(text)

        end_time = time.time()
        
        return {"summarization": text, 
                "elapsed_time": np.round(end_time-start_time,2), "article_id": article_id, 
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
    except Exception as e:
        print(f"Error occured during the Gemini model invokation: {e}")
        return None


    