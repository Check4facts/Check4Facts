
from check4facts.text_sum_scripts.translate import *
import numpy as np
from spacy.lang.el import Greek
nlp = Greek()
nlp.add_pipe("sentencizer")



def text_to_bulleted_list(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    html_bulleted_list = "<ul><br>" + "<br>".join(
        [
            f"<li>{sentence.replace('*', '')}</li>" 
            for sentence in sentences 
            if len(sentence.split()) >= 4 and sentence[-1].strip() in [';', '!', ',', '.', '"', '»', '>', '?']
        ]
    ) + "<br></ul>"

    return html_bulleted_list


def capitalize_bullets(text):
    lines = text.splitlines()
    updated_lines = []
    for line in lines:
        
        if line.strip().startswith("<li>") and line.strip().endswith("</li>"):
            start_tag = "<li>"
            end_tag = "</li>"
            content = line.strip()[len(start_tag):-len(end_tag)]
            
            if content:
                content = f"{content[0].upper()}{content[1:]}"
        
            updated_line = f"{start_tag}{content}{end_tag}"
        else:
            updated_line = line
        
        updated_lines.append(updated_line)
    
    return "".join(updated_lines)




