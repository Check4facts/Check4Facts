
from check4facts.scripts.text_sum.translate import *
import numpy as np
import re
from spacy.lang.el import Greek
# nlp = Greek()
# nlp.add_pipe("sentencizer")



def text_to_bullet_list(text):
    # doc = nlp(text)
    # sentences = [sent.text.strip() for sent in doc.sents]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    bulleted_list = "\n".join([f"• {sentence}" for sentence in sentences])
    return bulleted_list

def bullet_to_html_list(text):
    if isinstance(text, tuple): 
        text = text[0]
    elif not isinstance(text, str): 
        raise ValueError("Input must be a string or a tuple containing a string")
  
    bullet_pattern = r"^\s*[-•*]\s+"
    lines = re.findall(r'[^.]*\.+', text)
    html_lines = []
    
    for line in lines:
        if re.match(bullet_pattern, line):
            html_lines.append(f"<li>{line.strip('- •* ').strip()}</li><br>")
        else:
            html_lines.append(line)
            
    return "<ul>" + "".join(html_lines) + "</ul>"


def capitalize_bullets(text):
    return re.sub(r'(<li>)(.*?)(</li>)', lambda match: match.group(1) + match.group(2).capitalize() + match.group(3), text)




