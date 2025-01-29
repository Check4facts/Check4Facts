
from check4facts.scripts.text_sum.translate import *
import numpy as np
import re
from spacy.lang.el import Greek
nlp = Greek()
nlp.add_pipe("sentencizer")



def text_to_bullet_list(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    bulleted_list = "\n".join([f"• {sentence}" for sentence in sentences])
    return bulleted_list

def bullet_to_html_list(text):
    if isinstance(text, tuple):  # If it's a tuple, get the first element
        text = text[0]
    elif not isinstance(text, str):  # Handle other unexpected types
        raise ValueError("Input must be a string or a tuple containing a string")
    # Split the input text by '•' to get individual bullet points
    bullet_points = re.split(r'[•*-]\s*', text)
    
    # Remove any leading or trailing spaces from each bullet point
    bullet_points = [point.strip() for point in bullet_points if point.strip()]

    
    # Generate the HTML format for the bulleted list
    html_list = "<ul>"
    for point in bullet_points:
        if 'μια περίληψη' not in point:
            html_list += f"<li>{point}</li><br>"
    html_list += "</ul>"
    
    return html_list


def capitalize_bullets(text):
    return re.sub(r'(<li>)(.*?)(</li>)', lambda match: match.group(1) + match.group(2).capitalize() + match.group(3), text)




