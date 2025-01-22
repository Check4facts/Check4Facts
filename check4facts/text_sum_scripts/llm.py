# # script that runs a locally installed llm for summarization tasks
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# import torch
# import re
# import time
# from translate import *
# import numpy as np
# import spacy
# from spacy.lang.el import Greek
# nlp = Greek()
# nlp.add_pipe("sentencizer")
# #nlp = spacy.load("el_core_news_lg")
# #stanza.download('el')
# #nlp = stanza.Pipeline('el', processors='tokenize')


# def initialize_model():
#     model_name = "google-t5/t5-3b"
#     device = 0 if torch.cuda.is_available() else -1  
#     model = AutoModelForSeq2SeqLM.from_pretrained(
#         model_name, 
#         torch_dtype=torch.float16  
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer.model_max_length = 1024   
#     model.to(device)
#     summarization_pipeline = pipeline("summarization", model=model, 
#                                       tokenizer=tokenizer, device=device, 
#                                       max_length=514, truncation=True)    
#     return summarization_pipeline


# def generate_summary(text):
#     summarizer = initialize_model()
#     summary = summarizer(text, num_return_sequences=1)
#     return summary[0]['summary_text']

# def text_to_bulleted_list(text):
#     doc = nlp(text)
#     # Extract sentences from the text
#     sentences = [sent.text.strip() for sent in doc.sents]

#     # Generate HTML list items
#     html_bulleted_list = "<ul><br>" + "<br>".join(
#         [
#             f"<li>{sentence.replace('*', '')}</li>" 
#             for sentence in sentences 
#             if len(sentence.split()) >= 4 and sentence[-1].strip() in [';', '!', ',', '.', '"', '»', '>', '?']
#         ]
#     ) + "<br></ul>"

#     return html_bulleted_list


# def capitalize_bullets(text):
#     lines = text.splitlines()
#     updated_lines = []
#     for line in lines:
        
#         if line.strip().startswith("<li>") and line.strip().endswith("</li>"):
#             start_tag = "<li>"
#             end_tag = "</li>"
#             content = line.strip()[len(start_tag):-len(end_tag)]
            
#             if content:
#                 content = f"{content[0].upper()}{content[1:]}"
        
#             updated_line = f"{start_tag}{content}{end_tag}"
#         else:
#             updated_line = line
        
#         updated_lines.append(updated_line)
    
#     return "".join(updated_lines)

# def run(text):

    
    
#     start_time = time.time()
#     #text = df.loc[df['article_id'] == i, 'content'].values[0]
#     text = translate_long_text(text, src_lang='el', target_lang='en')
#     summary = generate_summary(text)
#     end_time = time.time()

#     #store the results
#     #label = df.loc[df['article_id'] == i, 'fact_checker_accuracy'].values[0]
    
    
#     #result_sum = f"Η δήλωση είναι {label}. "
#     #print(label)
#     result_sum = translate_long_text(summary,src_lang='en', target_lang='el')
#     elapsed_time = np.round(end_time-start_time,2)

#     bulleted_list = text_to_bulleted_list(result_sum)
#     bulleted_list = capitalize_bullets(bulleted_list)

#     #print(bulleted_list)
#     #print()
#     #print()
#     #print(elapsed_time)

#     return bulleted_list, elapsed_time



#run()
#####


