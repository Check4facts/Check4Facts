
from check4facts.scripts.text_sum.translate import *
import numpy as np
import re
from spacy.lang.el import Greek
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
#nltk.download('punkt')

# nlp = Greek()
# nlp.add_pipe("sentencizer")

def extract_text_from_html(html_string):
    soup = BeautifulSoup(html_string, 'html.parser')
    return soup.get_text(separator=' ', strip=True)

def text_to_bullet_list(text):

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    sentences = sent_tokenize(text, language='greek')
    bulleted_list = "\n".join([
    "• " + sentence.replace("*", "")
    for sentence in sentences 
    if "ακολουθεί μία περίληψη" not in sentence.lower()])

    return bulleted_list

def bullet_to_html_list(text):
    if isinstance(text, tuple): 
        text = text[0]
    elif not isinstance(text, str): 
        raise ValueError("Input must be a string or a tuple containing a string")

    bullet_points = re.split(r'\s*•\s*', text)
    bullet_points = [f"<li>{point.strip()}</li>" for point in bullet_points if point.strip()]


    return "<ul>" + "".join(bullet_points) + "</ul>"


def capitalize_bullets(text):
    return re.sub(r'(<li>)(.*?)(</li>)', lambda match: match.group(1) + match.group(2).capitalize() + match.group(3), text)




