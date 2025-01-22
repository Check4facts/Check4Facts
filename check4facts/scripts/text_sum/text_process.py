
from check4facts.scripts.text_sum.translate import *
import numpy as np
import re
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
    return re.sub(r'(<li>)(.*?)(</li>)', lambda match: match.group(1) + match.group(2).capitalize() + match.group(3), text)




